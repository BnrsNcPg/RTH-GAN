"""Resolved architecture/loss identities for prospective manuscript experiments."""
from __future__ import annotations


def intensity_control(config):
    thermal = config["thermal"]
    return thermal.get("control", thermal.get("weighting", "hcw"))


def experiment_identity(config):
    """Keep legacy labels while giving explicit ablations distinct result rows."""
    intensity = config["dataset"].get("input_mode", "quantitative_landsat") == "jpeg_intensity"
    model = config["model"]
    explicit = ("variant" in model or "fusion_variant" in model.get("generator", {})
                or "rpa_variant" in model.get("generator", {})
                or "control" in config["thermal"])
    if not explicit:
        return {"method": "rthgan_intensity_proxy" if intensity else "rthgan_manuscript",
                "method_display_name": "RTH-GAN (JPEG intensity proxy)" if intensity else "RTH-GAN (manuscript implementation)"}
    variant = model.get("variant", "rgg")
    fusion = model.get("generator", {}).get("fusion_variant", "full") if variant == "rgg" else "none"
    control = intensity_control(config) if intensity else "thermal_" + config["thermal"].get("weighting", "hcw")
    architecture = variant + ("_" + fusion if variant == "rgg" else "")
    rpa = model.get("generator", {}).get("rpa_variant", "full")
    if variant == "rgg" and rpa != "full":
        architecture += "_" + rpa
    identity = {"method": f"rthgan_{'jpeg' if intensity else 'quantitative'}_{architecture}_{control}",
            "method_display_name": f"{architecture} / {control}",
            "generator_variant": variant, "fusion_variant": fusion, "auxiliary_control": control}
    if "rpa_variant" in model.get("generator", {}):
        identity["rpa_variant"] = rpa
    if config.get("protocol") == "manuscript_reproducibility_v2":
        identity["method"] += "_repro_v2"
        identity["protocol"] = config["protocol"]
    return identity


def training_objectives(config):
    """Describe the actual trainer branches; validate explicit Stage-I claims."""
    settings = config["training"]
    thermal = config["thermal"]
    control = intensity_control(config)
    intensity = thermal.get("mode", "thermal") == "intensity_proxy"
    if intensity:
        terms = {"visual_only": None, "uniform": "uniform_proxy", "hcw": "hcw",
                 "clipping_matched_l1": "clipping_matched_l1"}
        if control not in terms:
            raise ValueError("Unknown intensity control in training objectives")
        auxiliary = terms[control]
    else:
        auxiliary = "thermal_" + control
    objective = "l1" if auxiliary is None else "l1+" + auxiliary
    declared = settings.get("stage_i_objective")
    if declared is not None and declared != objective:
        raise ValueError("stage_i_objective does not match the configured auxiliary control")
    generator_terms = {"l1": float(settings["lambda_l1"])}
    if auxiliary is not None:
        generator_terms[auxiliary] = float(settings["lambda_hac"])
    return {"schema_version": 1,
            "pinn_pretraining": {"separate_from_stage_i": True, "enabled": uses_pinn(config),
                                 "terms": ["pde", "boundary", "tv", "value"] if uses_pinn(config) else []},
            "stage_i": {"objective": objective, "generator_terms": generator_terms,
                        "discriminator_updates_per_batch": 0, "pinn_frozen": uses_pinn(config)},
            "stage_ii": {"generator_terms": {**generator_terms, "adversarial": 1.0},
                         "discriminator_updates_per_batch": int(settings["discriminator_steps"]),
                         "pinn_frozen": uses_pinn(config)}}


def uses_pinn(config):
    return config["model"].get("variant", "rgg") != "n"


def checkpoint_metadata(state):
    """Distinguish the weights being evaluated from the run's best selection."""
    position = state.get("model_state_position")
    source = "saved_model_state_position" if position is not None else "completed_epochs"
    if position is None:
        for phase in ("adversarial", "warmup", "pinn"):
            epoch = state.get("completed", {}).get(phase, 0)
            if epoch:
                position = {"phase": phase, "epoch": epoch}
                break
    selection = state.get("selection")
    is_selected = (position is not None and selection is not None
                   and all(position[key] == selection[key] for key in ("phase", "epoch")))
    return {"checkpoint_position": position, "checkpoint_position_source": source if position else None,
            "best_validation_selection": selection, "selected_checkpoint": selection if is_selected else None}
