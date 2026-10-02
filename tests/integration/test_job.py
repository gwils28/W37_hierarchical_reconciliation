"""Bloc 3 de bout en bout : un run publie un artefact immuable ; une hiérarchie modifiée est refusée."""
import json

import pytest

from w37_reconciliation.data import build_hierarchy, make_region_channel_data
from w37_reconciliation.pipeline.job import run_job

pytestmark = pytest.mark.slow
LEVELS = {"region": "Region", "feuille": "Region/Canal"}


@pytest.fixture
def cfg(contract):
    cfg = json.loads(json.dumps(contract))               # copie profonde
    cfg["quality_gate"]["level_weights"] = {"feuille": 0.7, "region": 0.3}
    return cfg


def test_run_publishes_artifact_with_lineage(tmp_path, cfg):
    res = run_job(build_hierarchy(make_region_channel_data()), cfg, tmp_path, "demo@v1", LEVELS)
    assert res.report.passed and res.method == "mint_shrink"
    files = {p.name for p in res.published_to.iterdir()}
    assert files == {"part-0.csv", "_lineage.json", "_gate.json"}
    lineage = json.loads((res.published_to / "_lineage.json").read_text())
    assert {"run_id", "s_hash", "w_version", "method_used", "lib_versions"} <= lineage.keys()


def test_changed_hierarchy_without_version_bump_is_refused(tmp_path, cfg):
    run_job(build_hierarchy(make_region_channel_data()), cfg, tmp_path, "demo@v1", LEVELS)
    df = make_region_channel_data()
    df["Canal"] = df["Canal"].replace({"RHD": "CHR"})     # canal renommé dans le référentiel
    with pytest.raises(RuntimeError, match="sans bump"):
        run_job(build_hierarchy(df), cfg, tmp_path, "demo@v1", LEVELS)
