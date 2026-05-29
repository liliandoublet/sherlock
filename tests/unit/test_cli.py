from typer.testing import CliRunner

from sherlock.cli import app

runner = CliRunner()


def test_cli_help():
    """La CLI répond à --help sans erreur."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "sherlock" in result.output.lower()


def test_cli_info():
    """La commande info affiche la config."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Renaissance" in result.output
    assert "LFI" in result.output


def test_cli_clean_missing_file(tmp_path):
    """clean lève une erreur si le fichier n'existe pas."""
    result = runner.invoke(
        app,
        [
            "clean",
            str(tmp_path / "inexistant.csv"),
            str(tmp_path / "output.csv"),
        ],
    )
    assert result.exit_code == 1


def test_cli_dataset_help():
    """La sous-commande dataset répond à --help."""
    result = runner.invoke(app, ["dataset", "--help"])
    assert result.exit_code == 0
    assert "merge" in result.output
    assert "split" in result.output
