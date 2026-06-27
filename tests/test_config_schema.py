"""Tests for the structured-config registration."""

from __future__ import annotations

from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore

from style_transfer import config_schema

########################################
#             Registration             #
########################################


def test_register_stores_root_config_schema() -> None:
    # register() must store the root Config under the name config.yaml's
    # `defaults` list references ("config_schema"). A mutated name would hide
    # the schema from Hydra; a mutated node would register the wrong type --
    # both surface here without composing a full run.
    #
    # Importing `run` (anywhere in the suite) calls register() at import time,
    # permanently seeding the entry into the global ConfigStore singleton. Drop
    # it first so this test's own register() call is what must repopulate it --
    # otherwise a no-op or wrong-name register() would still pass on the
    # leftover entry.
    repo = ConfigStore.instance().repo
    repo.pop("config_schema.yaml", None)

    config_schema.register()

    assert "config_schema.yaml" in repo

    node = repo["config_schema.yaml"].node
    assert OmegaConf.get_type(node) is config_schema.Config
