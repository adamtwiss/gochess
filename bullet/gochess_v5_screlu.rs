/*
GoChess v5 NNUE — SCReLU activation variant.
Same architecture as gochess_v5.rs but uses SCReLU instead of CReLU.
SCReLU = clamp(x, 0, 1)² — better gradient flow during training.
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
const QA: i16 = 255;  // input quantization scale
const QB: i16 = 64;   // output weight quantization scale

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
    let dataset_path = "/workspace/data/test80-2024-02-feb-2tb7p.min-v2.v6.binpack";
    let superbatches = 200;
    let initial_lr = 0.001;
    let final_lr = 0.0001;
    let wdl_proportion = 0.0;  // pure score training
    let save_rate = 50;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .use_threads(16)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        // NO .transpose() on weights — column-major "free transpose"
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i32>(QA as i32 * QB as i32),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", 768 * 16, HIDDEN_SIZE);
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, NUM_OUTPUT_BUCKETS);

            // SCReLU activation instead of CReLU
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden).select(output_buckets)
        });

    let schedule = TrainingSchedule {
        net_id: "gochess-v5-screlu".to_string(),
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
        threads: 16,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 128,
    };

    let dataloader = SfBinpackLoader::new(dataset_path, 256, 16, |entry| {
        entry.score.unsigned_abs() < 10000
    });

    trainer.run(&schedule, &settings, &dataloader);
}
