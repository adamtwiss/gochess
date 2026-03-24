/*
GoChess v7 NNUE — 1024 FT + 16→32 hidden layers, SCReLU.
(768×16 → 1024)×2 → 16 → 32 → 1×8

SCReLU activation on FT, L1, and L2 boundaries. wdl=0.0.
100 SB with LR warmup: 5 SB linear ramp 0.0001→0.001, then 95 SB cosine decay 0.001→0.0001.
Warmup prevents hidden layer collapse observed with plain cosine/100.
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

const FT_SIZE: usize = 1024;
const L1_SIZE: usize = 16;
const L2_SIZE: usize = 32;
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

    // === DATA FILES ===
    let data_files: Vec<&str> = vec![
        "/workspace/data/test80-2024-01-jan-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-02-feb-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-03-mar-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-04-apr-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-05-may-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-06-jun-2tb7p.min-v2.v6.binpack",
    ];

    let superbatches = 100;
    let save_rate = 10;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            // l0: Feature transformer weights and biases
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            // l1: First hidden layer weights and biases
            SavedFormat::id("l1w").round().quantise::<i16>(QA),
            SavedFormat::id("l1b").round().quantise::<i16>(QA),
            // l2: Second hidden layer weights and biases
            SavedFormat::id("l2w").round().quantise::<i16>(QA),
            SavedFormat::id("l2b").round().quantise::<i16>(QA),
            // l3: Output weights and biases
            SavedFormat::id("l3w").round().quantise::<i16>(QB),
            SavedFormat::id("l3b").round().quantise::<i32>(QA as i32 * QB as i32),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // Feature transformer: 768*16 -> 1024 per perspective
            let l0 = builder.new_affine("l0", 768 * 16, FT_SIZE);
            // Hidden layer 1: 2*1024 -> 16
            let l1 = builder.new_affine("l1", 2 * FT_SIZE, L1_SIZE);
            // Hidden layer 2: 16 -> 32
            let l2 = builder.new_affine("l2", L1_SIZE, L2_SIZE);
            // Output: 32 -> 8 buckets
            let l3 = builder.new_affine("l3", L2_SIZE, NUM_OUTPUT_BUCKETS);

            let stm = l0.forward(stm_inputs).screlu();
            let ntm = l0.forward(ntm_inputs).screlu();
            let h1 = l1.forward(stm.concat(ntm)).screlu();
            let h2 = l2.forward(h1).screlu();
            l3.forward(h2).select(output_buckets)
        });

    // LR schedule: 5 SB warmup (linear ramp 0.0001→0.001), then 95 SB cosine decay (0.001→0.0001)
    let schedule = TrainingSchedule {
        net_id: "gochess-v7-1024h16x32s-w0-e100w".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::Sequence {
            first: lr::LinearDecayLR {
                initial_lr: 0.0001,
                final_lr: 0.001,
                final_superbatch: 5,
            },
            second: lr::CosineDecayLR {
                initial_lr: 0.001,
                final_lr: 0.0001,
                final_superbatch: 95,
            },
            first_scheduler_final_superbatch: 5,
        },
        save_rate,
    };

    let settings = LocalSettings {
        threads: 32,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 512,
    };

    let dataloader = SfBinpackLoader::new_concat_multiple(
        &data_files, 512, 32, |entry| {
            entry.score.unsigned_abs() < 10000
        }
    );

    trainer.run(&schedule, &settings, &dataloader);
}
