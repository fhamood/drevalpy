"""Root ``drevalpy`` command (full experiment pipeline)."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli._helpers import as_list, pipeline_namespace
from drevalpy.utils import check_arguments, main


def register_pipeline_callback(app: typer.Typer) -> None:
    """Register the default callback that runs the full pipeline when no subcommand is given."""

    @app.callback(invoke_without_command=True)
    def pipeline_root(
        ctx: typer.Context,
        run_id: Annotated[str, typer.Option("--run_id")] = "my_run",
        path_data: Annotated[str, typer.Option("--path_data")] = "data",
        models: Annotated[list[str] | None, typer.Option("--models")] = None,
        baselines: Annotated[list[str] | None, typer.Option("--baselines")] = None,
        test_mode: Annotated[list[str] | None, typer.Option("--test_mode")] = None,
        randomization_mode: Annotated[list[str] | None, typer.Option("--randomization_mode")] = None,
        randomization_type: Annotated[str, typer.Option("--randomization_type")] = "permutation",
        n_trials_robustness: Annotated[int, typer.Option("--n_trials_robustness")] = 0,
        dataset_name: Annotated[str, typer.Option("--dataset_name")] = "GDSC1",
        cross_study_datasets: Annotated[list[str] | None, typer.Option("--cross_study_datasets")] = None,
        path_out: Annotated[str, typer.Option("--path_out")] = "results/",
        no_refitting: Annotated[bool, typer.Option("--no_refitting")] = False,
        curve_curator_cores: Annotated[int, typer.Option("--curve_curator_cores")] = 1,
        curve_curator_normalize: Annotated[bool, typer.Option("--curve_curator_normalize")] = False,
        measure: Annotated[str, typer.Option("--measure")] = "LN_IC50",
        overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
        optim_metric: Annotated[str, typer.Option("--optim_metric")] = "RMSE",
        wandb_project: Annotated[str | None, typer.Option("--wandb_project")] = None,
        n_cv_splits: Annotated[int, typer.Option("--n_cv_splits")] = 7,
        response_transformation: Annotated[str, typer.Option("--response_transformation")] = "None",
        multiprocessing: Annotated[bool, typer.Option("--multiprocessing")] = False,
        model_checkpoint_dir: Annotated[str, typer.Option("--model_checkpoint_dir")] = "TEMPORARY",
        final_model_on_full_data: Annotated[bool, typer.Option("--final_model_on_full_data")] = False,
        no_hyperparameter_tuning: Annotated[bool, typer.Option("--no_hyperparameter_tuning")] = False,
    ) -> None:
        """Run the drug response prediction model test suite."""
        if ctx.invoked_subcommand is not None:
            return
        args = pipeline_namespace(
            run_id=run_id,
            path_data=path_data,
            models=as_list(models),
            baselines=as_list(baselines) if baselines is not None else None,
            test_mode=as_list(test_mode) if test_mode is not None else ["LPO"],
            randomization_mode=as_list(randomization_mode) if randomization_mode is not None else ["None"],
            randomization_type=randomization_type,
            n_trials_robustness=n_trials_robustness,
            dataset_name=dataset_name,
            cross_study_datasets=as_list(cross_study_datasets),
            path_out=path_out,
            no_refitting=no_refitting,
            curve_curator_cores=curve_curator_cores,
            curve_curator_normalize=curve_curator_normalize,
            measure=measure,
            overwrite=overwrite,
            optim_metric=optim_metric,
            wandb_project=wandb_project,
            n_cv_splits=n_cv_splits,
            response_transformation=response_transformation,
            multiprocessing=multiprocessing,
            model_checkpoint_dir=model_checkpoint_dir,
            final_model_on_full_data=final_model_on_full_data,
            no_hyperparameter_tuning=no_hyperparameter_tuning,
        )
        check_arguments(args)
        main(args)
