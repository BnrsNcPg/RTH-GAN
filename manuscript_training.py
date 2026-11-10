"""Manuscript Algorithm 1: frozen priors, warm-up, alternating visual GAN."""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from models.discriminator import PatchDiscriminator
from models.manuscript_ablation import build_manuscript_generator
from models.thermal_rtd import build_thermal_auxiliary
from manuscript_protocol import training_objectives


@contextmanager
def frozen_discriminator(discriminator):
    flags = [parameter.requires_grad for parameter in discriminator.parameters()]
    training = discriminator.training
    discriminator.requires_grad_(False).eval()
    try:
        yield
    finally:
        for parameter, flag in zip(discriminator.parameters(), flags):
            parameter.requires_grad_(flag)
        discriminator.train(training)


class ManuscriptTrainer:
    def __init__(self, config, *, device="cpu"):
        self.config, self.device = config, torch.device(device)
        settings = config["training"]
        self.lambda_l1 = float(settings["lambda_l1"])
        self.lambda_hac = float(settings["lambda_hac"])
        self.discriminator_steps = settings["discriminator_steps"]
        if not 0 <= self.lambda_l1 < float("inf") or not 0 < self.lambda_hac < float("inf"):
            raise ValueError("Require finite lambda_l1 >= 0 and lambda_hac > 0")
        if type(self.discriminator_steps) is not int or self.discriminator_steps < 1:
            raise ValueError("discriminator_steps must be a positive integer")
        if config["model"]["name"] != "rthgan_manuscript":
            raise ValueError("This trainer only accepts rthgan_manuscript configurations")
        self.training_objectives = training_objectives(config)
        self.generator = build_manuscript_generator(config["model"]).to(device)
        self.discriminator = PatchDiscriminator(base_channels=config["model"]["discriminator_channels"]).to(device)
        self.auxiliary = build_thermal_auxiliary(config["thermal"]).to(device)
        betas = (float(settings.get("beta1", .5)), float(settings.get("beta2", .999)))
        self.optimizer_g = torch.optim.Adam([p for p in self.generator.parameters() if p.requires_grad], lr=settings["learning_rate_g"], betas=betas)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=settings["learning_rate_d"], betas=betas)
        self.optimizer_pinn = (torch.optim.Adam(self.generator.retinex.parameters(), lr=settings["learning_rate_pinn"])
                               if self.generator.retinex is not None else None)
        self.adversarial = nn.BCEWithLogitsLoss()

    def prepare_batch(self, batch):
        return {"visible": batch["visible"].to(self.device), "infrared": batch["infrared"].to(self.device),
                "context": {name: value.to(self.device) for name, value in batch.get("context", {}).items()}}

    @staticmethod
    def _backward(loss, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite manuscript training loss")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in parameters):
            raise FloatingPointError("Nonfinite manuscript parameter gradient")

    def pretrain_step(self, batch):
        if self.optimizer_pinn is None:
            raise ValueError("The N baseline has no PINN; set pinn_epochs=0")
        visible = batch["visible"].to(self.device)
        self.generator.retinex.requires_grad_(True).train()
        self.optimizer_pinn.zero_grad(set_to_none=True)
        losses = self.generator.retinex.loss_terms(visible)
        self._backward(losses.total, self.generator.retinex.parameters())
        self.optimizer_pinn.step()
        self.generator.retinex.zero_grad(set_to_none=True)
        self.generator.freeze_retinex()
        return {"pinn_total": float(losses.total.detach()), "pde": float(losses.pde.detach()),
                "boundary": float(losses.boundary.detach()), "tv": float(losses.tv.detach()), "value": float(losses.value.detach())}

    def train_step(self, batch, *, adversarial):
        batch = self.prepare_batch(batch)
        visible, target, context = batch["visible"], batch["infrared"], batch["context"]
        self.generator.train()
        self.generator.freeze_retinex()
        self.discriminator.train()
        discriminator_losses = []
        if adversarial:
            self.discriminator.requires_grad_(True)
            for _ in range(self.discriminator_steps):
                self.optimizer_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    detached_generated = self.generator(visible)
                real = self.discriminator(visible * 2 - 1, target)
                fake = self.discriminator(visible * 2 - 1, detached_generated)
                loss_d = self.adversarial(real, torch.ones_like(real)) + self.adversarial(fake, torch.zeros_like(fake))
                self._backward(loss_d, self.discriminator.parameters())
                self.optimizer_d.step()
                discriminator_losses.append(float(loss_d.detach()))
        self.optimizer_d.zero_grad(set_to_none=True)
        self.optimizer_g.zero_grad(set_to_none=True)
        with frozen_discriminator(self.discriminator):
            generated = self.generator(visible)
            l1 = (generated - target).abs().mean()
            thermal = self.auxiliary(generated, target, context)
            adversarial_loss = generated.new_zeros(())
            if adversarial:
                logits = self.discriminator(visible * 2 - 1, generated)
                adversarial_loss = self.adversarial(logits, torch.ones_like(logits))
            total = self.lambda_l1 * l1 + self.lambda_hac * thermal.loss + adversarial_loss
            self._backward(total, self.generator.parameters())
            self.optimizer_g.step()
        return {"g_total": float(total.detach()), "l1": float(l1.detach()),
                "hac": float(thermal.loss.detach()), "adversarial": float(adversarial_loss.detach()),
                "d_total": sum(discriminator_losses) / len(discriminator_losses) if discriminator_losses else 0.}

    @torch.no_grad()
    def validation_l1(self, loader):
        self.generator.eval()
        total, count = 0., 0
        for batch in loader:
            generated = self.generator(batch["visible"].to(self.device))
            errors = (generated - batch["infrared"].to(self.device)).abs().flatten(1).mean(1)
            total, count = total + float(errors.sum()), count + len(errors)
        if count == 0 or not torch.isfinite(torch.tensor(total)):
            raise ValueError("Validation requires nonempty finite paired predictions")
        return total / count

    def state_dict(self):
        return {"generator": self.generator.state_dict(), "discriminator": self.discriminator.state_dict(),
                "auxiliary": self.auxiliary.state_dict(), "optimizer_g": self.optimizer_g.state_dict(),
                "optimizer_d": self.optimizer_d.state_dict(),
                "optimizer_pinn": self.optimizer_pinn.state_dict() if self.optimizer_pinn else None}

    def load_state_dict(self, state):
        for name in ("generator", "discriminator", "auxiliary", "optimizer_g", "optimizer_d", "optimizer_pinn"):
            component = getattr(self, name)
            if component is None:
                if state[name] is not None:
                    raise ValueError(f"Checkpoint has an unexpected {name}")
            else:
                component.load_state_dict(state[name])
        self.generator.freeze_retinex()
