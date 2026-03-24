/*
GoChess v5 NNUE architecture — shallow wide net for Bullet GPU training.
(768×16 → 1024)×2 → 1×8

CReLU on the hidden layer, 8 output buckets by material count.
Single hidden layer — matches Bullet's recommended architecture.

Based on Alexandria's proven approach: wide accumulator, no depth.
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

const HIDDEN_SIZE: usize = 1024;
const NUM_OUTPUT_BUCKETS: usize = 8;
const SCALE: i32 = 400;
const QA: i16 = 255;  // input quantization scale (CReLU clips to [0, QA])
const QB: i16 = 64;   // output weight quantization scale

fn main() {
    // GoChess king bucket layout: mirroredFile * 4 + rankGroup
    // 16 buckets: 4 mirrored files × 4 rank groups
    #[rustfmt::skip]
    const BUCKET_LAYOUT: [usize; 32] = [
         0,  4,  8, 12,   // rank 0, files a-d
         0,  4,  8, 12,   // rank 1
         1,  5,  9, 13,   // rank 2
         1,  5,  9, 13,   // rank 3
         2,  6, 10, 14,   // rank 4
         2,  6, 10, 14,   // rank 5
         3,  7, 11, 15,   // rank 6
         3,  7, 11, 15,   // rank 7
    ];

    // === CONFIGURATION ===
    let data_files: Vec<&str> = vec![
        "/workspace/data/test80-2024-01-jan-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-02-feb-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-03-mar-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-04-apr-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-05-may-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-06-jun-2tb7p.min-v2.v6.binpack",
    ];
    let superbatches = 400;
    let initial_lr = 0.001;
    let final_lr = 0.0001;
    let wdl_proportion = 0.5;  // 50% game result, 50% score
    let save_rate = 20;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        // Quantization matching GoChess v5 inference
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i32>(QA as i32 * QB as i32),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // Single hidden layer: 768*16 -> 1024 per perspective
            let l0 = builder.new_affine("l0", 768 * 16, HIDDEN_SIZE);

            // Output: 2*1024 -> 8 buckets
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, NUM_OUTPUT_BUCKETS);

            // Inference: CReLU activation, dual perspective
            let stm_hidden = l0.forward(stm_inputs).crelu();
            let ntm_hidden = l0.forward(ntm_inputs).crelu();
            let hidden = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden).select(output_buckets)
        });

    let schedule = TrainingSchedule {
        net_id: "gochess-v5".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,  // ~100M positions per superbatch
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
        batch_queue_size: 512,
    };

    let dataloader = SfBinpackLoader::new_concat_multiple(&data_files, 256, 16, |entry| {
        entry.score.unsigned_abs() < 10000
    });

    trainer.run(&schedule, &settings, &dataloader);
}
