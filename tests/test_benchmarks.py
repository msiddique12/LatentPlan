from latent_plan.benchmarks import create_env, get_benchmark_specs


def test_benchmark_sets_non_empty() -> None:
    assert len(get_benchmark_specs("easy")) > 0
    assert len(get_benchmark_specs("hard")) > 0
    assert len(get_benchmark_specs("all")) >= len(get_benchmark_specs("easy"))


def test_create_env_matches_spec() -> None:
    spec = get_benchmark_specs("easy")[0]
    env = create_env(spec)
    assert env.width == spec.width
    assert env.height == spec.height
    assert env.start == spec.start
    assert env.goal == spec.goal
