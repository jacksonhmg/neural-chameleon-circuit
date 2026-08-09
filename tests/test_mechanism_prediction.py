from __future__ import annotations

import unittest

import torch

from neural_chameleon import (
    DiagnosticExample,
    LinearProbe,
    build_diagnostic_matrices,
    concept_leave_one_example_out_predictions,
    evaluate_diagnostic_predictions,
    fit_diagnostic_bases,
    fit_probe_standardization,
    fit_weighted_ridge,
    predict_original,
    response_position_features,
    select_alpha_leave_one_concept_out,
    validate_diagnostic_examples,
)


class MechanismPredictionTests(unittest.TestCase):
    def make_examples(self) -> tuple[list[DiagnosticExample], tuple[str, ...]]:
        generator = torch.Generator().manual_seed(31002)
        head_ids = ("layer_09.head_04", "layer_10.head_02")
        beta = torch.randn(16, 6, generator=generator)
        examples = []
        for concept_index, concept in enumerate(("a", "b", "c", "d")):
            for example_index in range(4):
                tokens = 5 + example_index
                first = (
                    torch.randn(tokens, 3, generator=generator) + concept_index * 0.1
                )
                second = (
                    torch.randn(tokens, 3, generator=generator) - concept_index * 0.05
                )
                normal = torch.randn(tokens, 6, generator=generator)
                position = response_position_features(tokens)[:, :4]
                predictors = torch.cat((first, second, normal, position), dim=1)
                target = predictors @ beta
                examples.append(
                    DiagnosticExample(
                        example_id=f"{concept}-{example_index}",
                        concept=concept,
                        writer={head_ids[0]: first, head_ids[1]: second},
                        normal_state=normal,
                        target_u=target,
                    )
                )
        return examples, head_ids

    def test_leakage_contract_bases_and_ridge_are_deterministic(self):
        examples, head_ids = self.make_examples()
        audit = validate_diagnostic_examples(examples, head_ids)
        self.assertEqual(
            audit["feature_fields"],
            [
                "observed_k12_delta",
                "normal_resid_post_8",
                "response_relative_position",
            ],
        )
        self.assertEqual(audit["forbidden_feature_fields_present"], [])
        first = fit_diagnostic_bases(
            examples,
            head_ids,
            writer_rank=3,
            normal_rank=6,
            target_rank=6,
            seed=31002,
        )
        second = fit_diagnostic_bases(
            examples,
            head_ids,
            writer_rank=3,
            normal_rank=6,
            target_rank=6,
            seed=31002,
        )
        self.assertTrue(torch.equal(first.target.components, second.target.components))
        for head_id in head_ids:
            self.assertTrue(
                torch.equal(
                    first.writer[head_id].components, second.writer[head_id].components
                )
            )

        matrices = build_diagnostic_matrices(examples, first)
        selection = select_alpha_leave_one_concept_out(
            matrices, first, (1e-6, 1e-4, 1e-2)
        )
        self.assertIn(selection.selected_alpha, (1e-6, 1e-4, 1e-2))
        fit = fit_weighted_ridge(
            matrices.full_features,
            matrices.target_coordinates,
            matrices.sample_weights,
            alpha=selection.selected_alpha,
        )
        prediction = predict_original(fit, matrices, first)
        probes = (
            LinearProbe(torch.randn(1, 6), torch.randn(1)),
            LinearProbe(torch.randn(1, 6), torch.randn(1)),
        )
        scale = fit_probe_standardization(examples, probes)
        metrics = evaluate_diagnostic_predictions(
            prediction, matrices, first, probes, scale
        )
        self.assertGreater(metrics.macro_r2_u, 0.99)
        self.assertLess(metrics.macro_probe_vector_snmse, 1e-3)

    def test_writer_features_add_value_over_normal_only(self):
        examples, head_ids = self.make_examples()
        bases = fit_diagnostic_bases(
            examples,
            head_ids,
            writer_rank=3,
            normal_rank=6,
            target_rank=6,
            seed=31002,
        )
        matrices = build_diagnostic_matrices(examples, bases)
        full = fit_weighted_ridge(
            matrices.full_features,
            matrices.target_coordinates,
            matrices.sample_weights,
            alpha=1e-6,
        )
        normal = fit_weighted_ridge(
            matrices.normal_features,
            matrices.target_coordinates,
            matrices.sample_weights,
            alpha=1e-6,
        )
        full_error = (
            (predict_original(full, matrices, bases) - matrices.target_original)
            .square()
            .mean()
        )
        normal_error = (
            (
                predict_original(normal, matrices, bases, normal_only=True)
                - matrices.target_original
            )
            .square()
            .mean()
        )
        self.assertLess(full_error, normal_error * 0.05)

    def test_concept_baseline_excludes_target_example(self):
        head_id = "layer_09.head_04"
        examples = [
            DiagnosticExample(
                example_id="one",
                concept="same",
                writer={head_id: torch.zeros(2, 1)},
                normal_state=torch.zeros(2, 2),
                target_u=torch.ones(2, 2),
            ),
            DiagnosticExample(
                example_id="three",
                concept="same",
                writer={head_id: torch.zeros(2, 1)},
                normal_state=torch.zeros(2, 2),
                target_u=torch.full((2, 2), 3.0),
            ),
        ]
        prediction = concept_leave_one_example_out_predictions(examples, torch.zeros(2))
        self.assertTrue(torch.equal(prediction[:2], torch.full((2, 2), 3.0)))
        self.assertTrue(torch.equal(prediction[2:], torch.ones(2, 2)))


if __name__ == "__main__":
    unittest.main()
