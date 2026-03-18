/*
GoChess v5 768 with pairwise multiplication.
(768×16 → 768)×2 with CReLU + pairwise_mul → 384×2 = 768 → 1×8

Smaller accumulator than 1024 baseline but with pairwise gating.
Pairwise mul halves the CReLU output: consecutive pairs are multiplied,
producing a learned gating mechanism. Output layer sees 384 per perspective.

768 accumulator = 25% fewer input params than 1024, but pairwise adds
expressiveness. Output layer is 768×8 vs 2048×8 = 62% smaller.
Should be faster than 1024 CReLU due to smaller accumulator + output.
*/
use bullet_lib::{
    game::{
        inputs::ChessBucketsMirrored,
        outputs::MaterialCount,
    },
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::SfBinpackLoader},
};

const HIDDEN_SIZE: usize = 768;  // pre-pairwise (384 after pairwise)
const NUM_OUTPUT_BUCKETS: usize = 8;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

fn main() {
    #[rustfmt::skip]
    const BUCKET_LAYOUT: [usize; 32] = [
         0,  4,  8, 12,
         0,  4,  8, 12,
         1,  5,  9, 13,
         1,  5,  9, 13,
         2,  6, 10, 14,
         2,  6, 10, 14,
         3,  7, 11, 15,
         3,  7, 11, 15,
    ];

    // === CONFIGURATION ===
    // UPDATE dataset_paths to actual paths on GPU server
    let dataset_paths = &[
        "/workspace/data/test80-2024-01.min-v2.v6.binpack",
        "/workspace/data/test80-2024-02.min-v2.v6.binpack",
        "/workspace/data/test80-2024-03.min-v2.v6.binpack",
    ];
    let superbatches = 200;
    let initial_lr = 0.001;
    let final_lr = 0.0001;
    let wdl_proportion = 0.5;  // 50% game result, 50% score — must match baseline
    let save_rate = 40;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            // Output weights: 384*2 = 768 inputs (after pairwise concat)
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i32>(QA as i32 * QB as i32),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", 768 * 16, HIDDEN_SIZE);
            // After pairwise: 384 per perspective, 768 concatenated
            let l1 = builder.new_affine("l1", HIDDEN_SIZE, NUM_OUTPUT_BUCKETS);

            // CReLU + pairwise_mul halves the accumulator
            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hidden = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden).select(output_buckets)
        });

    let schedule = TrainingSchedule {
        net_id: "gochess-v5-768pw".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_proportion },
        lr_scheduler: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: superbatches },
        save_rate,
    };

    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 64,
    };

    let dataloader = SfBinpackLoader::new(dataset_paths[0], 256, 16, |entry| {
        entry.score.unsigned_abs() < 10000
    });

    trainer.run(&schedule, &settings, &dataloader);
}
