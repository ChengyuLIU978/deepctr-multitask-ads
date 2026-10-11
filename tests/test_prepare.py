from ctr_cvr.data import generate_demo_data
from ctr_cvr.prepare_criteo import prepare


def test_prepared_sample_is_byte_reproducible(tmp_path):
    source = tmp_path / "source.tsv.gz"
    generate_demo_data(1_000, seed=5).to_csv(source, sep="\t", index=False, compression="gzip")
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    first_metadata = prepare(source, first, 0.5, 9, 200, "test://source")
    second_metadata = prepare(source, second, 0.5, 9, 200, "test://source")
    assert first_metadata["sampled_rows"] == second_metadata["sampled_rows"]
    assert first_metadata["prepared_sha256"] == second_metadata["prepared_sha256"]
    assert first_metadata["source_path"] == source.name

