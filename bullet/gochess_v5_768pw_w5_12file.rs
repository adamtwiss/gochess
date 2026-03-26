/*
GoChess v5 768pw — pairwise CReLU, wdl=0.05, 12 data files (2023+2024).
(768×16 → 768)×2 → pairwise → 1×8

Pairwise multiplication provides non-linearity (feature interaction)
without hidden layers. CReLU is standard for pairwise (SCReLU would
double the non-linearity — not what we want).

12 T80 files for maximum data diversity. wdl=0.05 for near-zero resolution.
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

const FT_SIZE: usize = 768;
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

    let data_files: Vec<&str> = vec![
        "/workspace/data/test80-2023-01-jan-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2023-02-feb-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2023-03-mar-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2023-04-apr-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2023-05-may-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2023-06-jun-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-01-jan-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-02-feb-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-03-mar-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-04-apr-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-05-may-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-06-jun-2tb7p.min-v2.v6.binpack",
    ];

    let superbatches = 800;
    let save_rate = 100;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i32>(QA as i32 * QB as i32),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", 768 * 16, FT_SIZE);
            let l1 = builder.new_affine("l1", FT_SIZE, NUM_OUTPUT_BUCKETS);

            let stm = l0.forward(stm_inputs).crelu();
            let ntm = l0.forward(ntm_inputs).crelu();
            let paired = stm.pairwise_mul(ntm);
            l1.forward(paired).select(output_buckets)
        });

    let schedule = TrainingSchedule {
        net_id: "gochess-v5-768pw-w5-12file".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.05 },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: 0.001,
            final_lr: 0.0001,
            final_superbatch: superbatches,
        },
        save_rate,
    };

    let settings = LocalSettings {
        threads: 4,
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
