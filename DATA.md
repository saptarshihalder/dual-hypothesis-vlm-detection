# Data Setup

## Required folder structure

    dataset/
      fake/
        midjourney/          <- Midjourney V5/V6 generated images (training fakes)
                                Any filename, .jpg or .png
      real/
        coco/                <- COCO 2017 train images (training reals)
                                Download: http://images.cocodataset.org/zips/train2017.zip
      cnndetection_test/     <- CNNDetection test set (evaluation only, never trained on)
        biggan/
          val/
            0_real/          <- real images
            1_fake/          <- biggan generated images
          test/
            0_real/
            1_fake/
        crn/      (same structure)
        cyclegan/ (same structure)
        deepfake/ (same structure)
        gaugan/   (same structure)
        imle/     (same structure)
        progan/   (same structure)
        san/      (same structure)
        seeingdark/ (same structure)
        stargan/  (same structure)
        stylegan/ (same structure)
        stylegan2/(same structure)
        whichfaceisreal/ (same structure)

## Downloading CNNDetection

    # From the original CNNDetection repo (Wang et al. CVPR 2020)
    # https://github.com/peterwang512/CNNDetection
    # Google Drive: https://drive.google.com/file/d/1z_fD3UKgWQyOTZIBbYSaQ-hz4AzUrLC1

    gdown 'https://drive.google.com/u/0/uc?id=1z_fD3UKgWQyOTZIBbYSaQ-hz4AzUrLC1' \
        -O CNN_synth_testset.zip
    unzip CNN_synth_testset.zip

## Split sizes (as used in this paper)

    Midjourney+COCO total items after teacher-prob filtering:
      Train : 70%
      Val   : 15%  (used for early stopping only)
      Test  : 15%  (MJ in-distribution test, AUROC=1.00)

    CNNDetection: 100% evaluation only
      Total test images: ~83,759 across 13 generators

## Setting paths

Export environment variables before training:

    export DATA_ROOT=/path/to/dataset
    export TEACHER_PROB=/path/to/teacher_probs.npz

Or edit lines 47-49 of scripts/training/train_dhsd_v2.py directly.
