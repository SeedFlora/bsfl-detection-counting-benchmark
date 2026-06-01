# Reviewer Experiment Results Summary

## Post-processing Sweep
| variant | mae_mean | mae_std | mae_min | mae_max | rmse_mean | rmse_std | rmse_min | rmse_max | r2_mean | r2_std | r2_min | r2_max | exact_match_rate_mean | exact_match_rate_std | exact_match_rate_min | exact_match_rate_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo11n | 6.1481 | 1.3921 | 5.3333 | 7.7556 | 16.1384 | 4.7440 | 13.2195 | 21.6122 | 0.8120 | 0.1133 | 0.6812 | 0.8807 | 0.3852 | 0.0925 | 0.3111 | 0.4889 |
| yolov8n | 4.2074 | 0.9801 | 3.3778 | 5.2889 | 9.6217 | 3.2148 | 7.1274 | 13.2497 | 0.9321 | 0.0456 | 0.8802 | 0.9653 | 0.3185 | 0.1283 | 0.2444 | 0.4667 |

## Robustness Sensitivity
| condition | variant | mae_mean | mae_std | mae_min | mae_max | rmse_mean | rmse_std | rmse_min | rmse_max | r2_mean | r2_std | r2_min | r2_max | exact_match_rate_mean | exact_match_rate_std | exact_match_rate_min | exact_match_rate_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brightness_bright | yolo11n | 7.6815 | 1.8336 | 5.6444 | 9.2000 | 17.8535 | 5.2873 | 12.2129 | 22.6970 | 0.7698 | 0.1250 | 0.6484 | 0.8982 | 0.3481 | 0.0257 | 0.3333 | 0.3778 |
| brightness_bright | yolov8n | 9.9037 | 1.3489 | 8.6222 | 11.3111 | 21.0660 | 3.6684 | 17.5322 | 24.8556 | 0.6910 | 0.1066 | 0.5784 | 0.7902 | 0.3556 | 0.0385 | 0.3333 | 0.4000 |
| brightness_dark | yolo11n | 8.2370 | 1.4889 | 6.6222 | 9.5556 | 19.9310 | 4.7820 | 15.2912 | 24.8435 | 0.7185 | 0.1317 | 0.5788 | 0.8404 | 0.3704 | 0.0257 | 0.3556 | 0.4000 |
| brightness_dark | yolov8n | 10.5407 | 1.3796 | 9.3333 | 12.0444 | 23.5560 | 3.9741 | 20.1274 | 27.9118 | 0.6141 | 0.1314 | 0.4683 | 0.7235 | 0.3556 | 0.0385 | 0.3333 | 0.4000 |
| contrast_high | yolo11n | 8.4222 | 1.4335 | 6.8444 | 9.6444 | 20.2862 | 4.1696 | 16.0499 | 24.3858 | 0.7112 | 0.1151 | 0.5942 | 0.8242 | 0.3111 | 0.0444 | 0.2667 | 0.3556 |
| contrast_high | yolov8n | 10.4593 | 1.3411 | 9.3111 | 11.9333 | 23.4311 | 3.6409 | 20.2402 | 27.3971 | 0.6193 | 0.1193 | 0.4878 | 0.7204 | 0.3704 | 0.0339 | 0.3333 | 0.4000 |
| contrast_low | yolo11n | 7.1111 | 1.6610 | 5.2667 | 8.4889 | 16.8781 | 4.9716 | 11.7540 | 21.6815 | 0.7943 | 0.1133 | 0.6792 | 0.9057 | 0.3630 | 0.0128 | 0.3556 | 0.3778 |
| contrast_low | yolov8n | 9.4815 | 1.6927 | 7.9333 | 11.2889 | 20.6737 | 4.2986 | 16.6447 | 25.1988 | 0.6999 | 0.1236 | 0.5667 | 0.8109 | 0.3630 | 0.0513 | 0.3333 | 0.4222 |
| gaussian_blur | yolov8n | 4.3259 | 0.6051 | 3.6444 | 4.8000 | 9.2581 | 1.7285 | 7.3937 | 10.8074 | 0.9401 | 0.0213 | 0.9203 | 0.9627 | 0.3704 | 0.0559 | 0.3111 | 0.4222 |
| gaussian_blur | yolo11n | 7.5333 | 3.3113 | 5.1333 | 11.3111 | 16.1575 | 7.4710 | 11.0040 | 24.7256 | 0.7964 | 0.1856 | 0.5828 | 0.9174 | 0.2963 | 0.0257 | 0.2667 | 0.3111 |
| gaussian_noise | yolo11n | 7.6519 | 1.3282 | 6.7556 | 9.1778 | 20.4443 | 4.2988 | 17.8885 | 25.4073 | 0.7064 | 0.1272 | 0.5595 | 0.7816 | 0.3704 | 0.0128 | 0.3556 | 0.3778 |
| gaussian_noise | yolov8n | 12.0296 | 2.3165 | 9.3556 | 13.4222 | 26.6206 | 4.0405 | 21.9671 | 29.2385 | 0.5090 | 0.1405 | 0.4166 | 0.6707 | 0.2519 | 0.0128 | 0.2444 | 0.2667 |
| original | yolo11n | 7.9926 | 1.7054 | 6.0667 | 9.3111 | 19.3692 | 4.8938 | 14.2696 | 24.0273 | 0.7331 | 0.1275 | 0.6060 | 0.8610 | 0.3259 | 0.0339 | 0.2889 | 0.3556 |
| original | yolov8n | 10.0815 | 1.5461 | 8.5778 | 11.6667 | 22.3607 | 4.1935 | 18.2611 | 26.6421 | 0.6508 | 0.1289 | 0.5156 | 0.7724 | 0.3481 | 0.0257 | 0.3333 | 0.3778 |
| synthetic_occlusion | yolo11n | 7.7556 | 1.4658 | 6.0889 | 8.8444 | 19.1880 | 4.6936 | 14.1516 | 23.4398 | 0.7387 | 0.1195 | 0.6250 | 0.8633 | 0.3185 | 0.0559 | 0.2667 | 0.3778 |
| synthetic_occlusion | yolov8n | 9.3259 | 1.7670 | 7.5778 | 11.1111 | 20.9260 | 4.7102 | 16.1087 | 25.5212 | 0.6911 | 0.1337 | 0.5555 | 0.8229 | 0.3185 | 0.0257 | 0.2889 | 0.3333 |

## Training Ablations
| case | variant | test_count_mae_mean | test_count_mae_std | test_count_mae_min | test_count_mae_max | test_count_rmse_mean | test_count_rmse_std | test_count_rmse_min | test_count_rmse_max | test_count_r2_mean | test_count_r2_std | test_count_r2_min | test_count_r2_max | test_detection_mAP50_mean | test_detection_mAP50_std | test_detection_mAP50_min | test_detection_mAP50_max | test_detection_mAP50_95_mean | test_detection_mAP50_95_std | test_detection_mAP50_95_min | test_detection_mAP50_95_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minimal_aug20 | yolo11n | 4.2741 | 0.6005 | 3.8222 | 4.9556 | 11.9776 | 3.3425 | 9.8342 | 15.8289 | 0.8970 | 0.0590 | 0.8290 | 0.9340 | 0.8147 | 0.0079 | 0.8100 | 0.8238 | 0.6484 | 0.0095 | 0.6403 | 0.6589 |
| minimal_aug20 | yolov8n | 4.7778 | 0.6900 | 4.1111 | 5.4889 | 12.6496 | 2.1993 | 10.7693 | 15.0680 | 0.8886 | 0.0391 | 0.8451 | 0.9209 | 0.8245 | 0.0023 | 0.8219 | 0.8263 | 0.6588 | 0.0015 | 0.6570 | 0.6598 |
| default20 | yolo11n | 7.9926 | 1.7054 | 6.0667 | 9.3111 | 19.3692 | 4.8938 | 14.2696 | 24.0273 | 0.7331 | 0.1275 | 0.6060 | 0.8610 | 0.8479 | 0.0030 | 0.8451 | 0.8511 | 0.6823 | 0.0023 | 0.6802 | 0.6848 |
| default50 | yolo11n | 9.2148 | 1.1778 | 8.0444 | 10.4000 | 19.8421 | 2.5411 | 17.2098 | 22.2810 | 0.7284 | 0.0684 | 0.6612 | 0.7979 | 0.8686 | 0.0004 | 0.8682 | 0.8691 | 0.7172 | 0.0031 | 0.7142 | 0.7204 |
| robust_aug20 | yolov8n | 9.6148 | 1.2197 | 8.2889 | 10.6889 | 22.8303 | 1.7476 | 20.8129 | 23.8821 | 0.6429 | 0.0533 | 0.6108 | 0.7044 | 0.8494 | 0.0094 | 0.8427 | 0.8602 | 0.6796 | 0.0047 | 0.6754 | 0.6847 |
| default20 | yolov8n | 10.0815 | 1.5461 | 8.5778 | 11.6667 | 22.3607 | 4.1935 | 18.2611 | 26.6421 | 0.6508 | 0.1289 | 0.5156 | 0.7724 | 0.8600 | 0.0096 | 0.8491 | 0.8671 | 0.6966 | 0.0097 | 0.6855 | 0.7023 |
| robust_aug20 | yolo11n | 11.7259 | 0.8322 | 11.0222 | 12.6444 | 25.3928 | 2.5023 | 23.2169 | 28.1271 | 0.5571 | 0.0881 | 0.4601 | 0.6321 | 0.8587 | 0.0058 | 0.8529 | 0.8644 | 0.6797 | 0.0069 | 0.6720 | 0.6854 |
| default50 | yolov8n | 14.3333 | 0.7327 | 13.8667 | 15.1778 | 31.3934 | 1.2741 | 29.9518 | 32.3684 | 0.3267 | 0.0541 | 0.2850 | 0.3878 | 0.8584 | 0.0015 | 0.8567 | 0.8597 | 0.7094 | 0.0035 | 0.7058 | 0.7129 |

## Paired Statistical Tests
| context | comparison | metric | left | right | n_pairs | left_mean | right_mean | mean_difference_left_minus_right | std_difference | paired_t_statistic | paired_t_p | wilcoxon_or_sign_p | cohens_dz | rank_biserial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| postprocessing_selected_test_by_seed | yolov8n vs yolo11n | mae | yolov8n | yolo11n | 3 | 4.2074 | 6.1481 | -1.9407 | 0.5454 | -6.1634 | 0.0253 | 0.2500 | -3.5585 | -1.0000 |
| postprocessing_selected_test_by_seed | yolov8n vs yolo11n | rmse | yolov8n | yolo11n | 3 | 9.6217 | 16.1384 | -6.5167 | 1.8162 | -6.2148 | 0.0249 | 0.2500 | -3.5881 | -1.0000 |
| postprocessing_selected_test_by_seed | yolov8n vs yolo11n | r2 | yolov8n | yolo11n | 3 | 0.9321 | 0.8120 | 0.1201 | 0.0691 | 3.0102 | 0.0949 | 0.2500 | 1.7379 | 1.0000 |
| robustness_brightness_bright_by_seed | yolov8n vs yolo11n | mae | yolov8n | yolo11n | 3 | 9.9037 | 7.6815 | 2.2222 | 1.8580 | 2.0715 | 0.1741 | 0.2500 | 1.1960 | 1.0000 |
| robustness_brightness_bright_by_seed | yolov8n vs yolo11n | rmse | yolov8n | yolo11n | 3 | 21.0660 | 17.8535 | 3.2125 | 4.9429 | 1.1257 | 0.3772 | 0.5000 | 0.6499 | 0.6667 |
| robustness_brightness_bright_by_seed | yolov8n vs yolo11n | r2 | yolov8n | yolo11n | 3 | 0.6910 | 0.7698 | -0.0787 | 0.1109 | -1.2292 | 0.3440 | 0.5000 | -0.7097 | -0.6667 |
| robustness_brightness_dark_by_seed | yolov8n vs yolo11n | mae | yolov8n | yolo11n | 3 | 10.5407 | 8.2370 | 2.3037 | 1.4202 | 2.8096 | 0.1068 | 0.2500 | 1.6221 | 1.0000 |
| robustness_brightness_dark_by_seed | yolov8n vs yolo11n | rmse | yolov8n | yolo11n | 3 | 23.5560 | 19.9310 | 3.6250 | 3.4679 | 1.8105 | 0.2119 | 0.2500 | 1.0453 | 1.0000 |
| robustness_brightness_dark_by_seed | yolov8n vs yolo11n | r2 | yolov8n | yolo11n | 3 | 0.6141 | 0.7185 | -0.1044 | 0.0887 | -2.0372 | 0.1785 | 0.2500 | -1.1762 | -1.0000 |
| robustness_contrast_high_by_seed | yolov8n vs yolo11n | mae | yolov8n | yolo11n | 3 | 10.4593 | 8.4222 | 2.0370 | 1.3949 | 2.5293 | 0.1272 | 0.2500 | 1.4603 | 1.0000 |
| robustness_contrast_high_by_seed | yolov8n vs yolo11n | rmse | yolov8n | yolo11n | 3 | 23.4311 | 20.2862 | 3.1449 | 3.3962 | 1.6039 | 0.2499 | 0.5000 | 0.9260 | 0.6667 |
| robustness_contrast_high_by_seed | yolov8n vs yolo11n | r2 | yolov8n | yolo11n | 3 | 0.6193 | 0.7112 | -0.0919 | 0.0906 | -1.7569 | 0.2210 | 0.5000 | -1.0143 | -0.6667 |
| robustness_contrast_low_by_seed | yolov8n vs yolo11n | mae | yolov8n | yolo11n | 3 | 9.4815 | 7.1111 | 2.3704 | 1.8381 | 2.2337 | 0.1551 | 0.2500 | 1.2896 | 1.0000 |
| robustness_contrast_low_by_seed | yolov8n vs yolo11n | rmse | yolov8n | yolo11n | 3 | 20.6737 | 16.8781 | 3.7955 | 4.4954 | 1.4624 | 0.2811 | 0.5000 | 0.8443 | 0.6667 |
| robustness_contrast_low_by_seed | yolov8n vs yolo11n | r2 | yolov8n | yolo11n | 3 | 0.6999 | 0.7943 | -0.0944 | 0.0994 | -1.6451 | 0.2417 | 0.5000 | -0.9498 | -0.6667 |
| robustness_gaussian_blur_by_seed | yolov8n vs yolo11n | mae | yolov8n | yolo11n | 3 | 4.3259 | 7.5333 | -3.2074 | 3.8803 | -1.4317 | 0.2886 | 0.2500 | -0.8266 | -1.0000 |
| robustness_gaussian_blur_by_seed | yolov8n vs yolo11n | rmse | yolov8n | yolo11n | 3 | 9.2581 | 16.1575 | -6.8994 | 9.0383 | -1.3222 | 0.3171 | 0.2500 | -0.7633 | -1.0000 |
| robustness_gaussian_blur_by_seed | yolov8n vs yolo11n | r2 | yolov8n | yolo11n | 3 | 0.9401 | 0.7964 | 0.1437 | 0.2046 | 1.2163 | 0.3479 | 0.2500 | 0.7022 | 1.0000 |
| robustness_gaussian_noise_by_seed | yolov8n vs yolo11n | mae | yolov8n | yolo11n | 3 | 12.0296 | 7.6519 | 4.3778 | 2.1143 | 3.5864 | 0.0697 | 0.2500 | 2.0706 | 1.0000 |
| robustness_gaussian_noise_by_seed | yolov8n vs yolo11n | rmse | yolov8n | yolo11n | 3 | 26.6206 | 20.4443 | 6.1763 | 3.9765 | 2.6902 | 0.1149 | 0.2500 | 1.5532 | 1.0000 |
| robustness_gaussian_noise_by_seed | yolov8n vs yolo11n | r2 | yolov8n | yolo11n | 3 | 0.5090 | 0.7064 | -0.1974 | 0.1265 | -2.7026 | 0.1140 | 0.2500 | -1.5604 | -1.0000 |
| robustness_original_by_seed | yolov8n vs yolo11n | mae | yolov8n | yolo11n | 3 | 10.0815 | 7.9926 | 2.0889 | 1.9912 | 1.8170 | 0.2109 | 0.5000 | 1.0491 | 0.6667 |
| robustness_original_by_seed | yolov8n vs yolo11n | rmse | yolov8n | yolo11n | 3 | 22.3607 | 19.3692 | 2.9915 | 4.7408 | 1.0929 | 0.3885 | 0.5000 | 0.6310 | 0.6667 |
| robustness_original_by_seed | yolov8n vs yolo11n | r2 | yolov8n | yolo11n | 3 | 0.6508 | 0.7331 | -0.0823 | 0.1187 | -1.2008 | 0.3528 | 0.5000 | -0.6933 | -0.6667 |
| robustness_synthetic_occlusion_by_seed | yolov8n vs yolo11n | mae | yolov8n | yolo11n | 3 | 9.3259 | 7.7556 | 1.5704 | 2.0677 | 1.3155 | 0.3189 | 0.5000 | 0.7595 | 0.6667 |
| robustness_synthetic_occlusion_by_seed | yolov8n vs yolo11n | rmse | yolov8n | yolo11n | 3 | 20.9260 | 19.1880 | 1.7380 | 5.4385 | 0.5535 | 0.6355 | 0.7500 | 0.3196 | 0.3333 |
| robustness_synthetic_occlusion_by_seed | yolov8n vs yolo11n | r2 | yolov8n | yolo11n | 3 | 0.6911 | 0.7387 | -0.0476 | 0.1332 | -0.6196 | 0.5987 | 0.7500 | -0.3577 | -0.3333 |
| robustness_degradation_yolo11n_by_seed | brightness_bright vs original | mae | brightness_bright | original | 3 | 7.6815 | 7.9926 | -0.3111 | 0.1736 | -3.1047 | 0.0900 | 0.2500 | -1.7925 | -1.0000 |
| robustness_degradation_yolo11n_by_seed | brightness_dark vs original | mae | brightness_dark | original | 3 | 8.2370 | 7.9926 | 0.2444 | 0.3111 | 1.3609 | 0.3066 | 0.5000 | 0.7857 | 0.6667 |
| robustness_degradation_yolo11n_by_seed | contrast_high vs original | mae | contrast_high | original | 3 | 8.4222 | 7.9926 | 0.4296 | 0.3114 | 2.3898 | 0.1394 | 0.2500 | 1.3798 | 1.0000 |
| robustness_degradation_yolo11n_by_seed | contrast_low vs original | mae | contrast_low | original | 3 | 7.1111 | 7.9926 | -0.8815 | 0.1224 | -12.4746 | 0.0064 | 0.2500 | -7.2022 | -1.0000 |
| robustness_degradation_yolo11n_by_seed | gaussian_blur vs original | mae | gaussian_blur | original | 3 | 7.5333 | 7.9926 | -0.4593 | 5.0150 | -0.1586 | 0.8885 | 1.0000 | -0.0916 | 0.0000 |
| robustness_degradation_yolo11n_by_seed | gaussian_noise vs original | mae | gaussian_noise | original | 3 | 7.6519 | 7.9926 | -0.3407 | 1.4115 | -0.4181 | 0.7165 | 0.7500 | -0.2414 | -0.3333 |
| robustness_degradation_yolo11n_by_seed | synthetic_occlusion vs original | mae | synthetic_occlusion | original | 3 | 7.7556 | 7.9926 | -0.2370 | 0.2458 | -1.6704 | 0.2368 | 0.5000 | -0.9644 | -0.6667 |
| robustness_degradation_yolov8n_by_seed | brightness_bright vs original | mae | brightness_bright | original | 3 | 9.9037 | 10.0815 | -0.1778 | 0.2037 | -1.5119 | 0.2697 | 0.5000 | -0.8729 | -0.6667 |
| robustness_degradation_yolov8n_by_seed | brightness_dark vs original | mae | brightness_dark | original | 3 | 10.5407 | 10.0815 | 0.4593 | 0.2651 | 3.0004 | 0.0954 | 0.2500 | 1.7323 | 1.0000 |
| robustness_degradation_yolov8n_by_seed | contrast_high vs original | mae | contrast_high | original | 3 | 10.4593 | 10.0815 | 0.3778 | 0.3151 | 2.0769 | 0.1734 | 0.2500 | 1.1991 | 1.0000 |
| robustness_degradation_yolov8n_by_seed | contrast_low vs original | mae | contrast_low | original | 3 | 9.4815 | 10.0815 | -0.6000 | 0.2037 | -5.1025 | 0.0363 | 0.2500 | -2.9459 | -1.0000 |
| robustness_degradation_yolov8n_by_seed | gaussian_blur vs original | mae | gaussian_blur | original | 3 | 4.3259 | 10.0815 | -5.7556 | 1.7564 | -5.6758 | 0.0297 | 0.2500 | -3.2769 | -1.0000 |
| robustness_degradation_yolov8n_by_seed | gaussian_noise vs original | mae | gaussian_noise | original | 3 | 12.0296 | 10.0815 | 1.9481 | 2.6941 | 1.2525 | 0.3370 | 0.5000 | 0.7231 | 0.6667 |
| robustness_degradation_yolov8n_by_seed | synthetic_occlusion vs original | mae | synthetic_occlusion | original | 3 | 9.3259 | 10.0815 | -0.7556 | 0.2255 | -5.8026 | 0.0284 | 0.2500 | -3.3501 | -1.0000 |
| training_default20_by_seed | yolov8n vs yolo11n | test_count_mae | yolov8n | yolo11n | 3 | 10.0815 | 7.9926 | 2.0889 | 1.9912 | 1.8170 | 0.2109 | 0.5000 | 1.0491 | 0.6667 |
| training_default20_by_seed | yolov8n vs yolo11n | test_count_rmse | yolov8n | yolo11n | 3 | 22.3607 | 19.3692 | 2.9915 | 4.7408 | 1.0929 | 0.3885 | 0.5000 | 0.6310 | 0.6667 |
| training_default20_by_seed | yolov8n vs yolo11n | test_count_r2 | yolov8n | yolo11n | 3 | 0.6508 | 0.7331 | -0.0823 | 0.1187 | -1.2008 | 0.3528 | 0.5000 | -0.6933 | -0.6667 |
| training_default20_by_seed | yolov8n vs yolo11n | test_detection_mAP50 | yolov8n | yolo11n | 3 | 0.8600 | 0.8479 | 0.0121 | 0.0102 | 2.0570 | 0.1760 | 0.2500 | 1.1876 | 1.0000 |
| training_default50_by_seed | yolov8n vs yolo11n | test_count_mae | yolov8n | yolo11n | 3 | 14.3333 | 9.2148 | 5.1185 | 0.6886 | 12.8738 | 0.0060 | 0.2500 | 7.4327 | 1.0000 |
| training_default50_by_seed | yolov8n vs yolo11n | test_count_rmse | yolov8n | yolo11n | 3 | 31.3934 | 19.8421 | 11.5513 | 2.6852 | 7.4511 | 0.0175 | 0.2500 | 4.3019 | 1.0000 |
| training_default50_by_seed | yolov8n vs yolo11n | test_count_r2 | yolov8n | yolo11n | 3 | 0.3267 | 0.7284 | -0.4017 | 0.0793 | -8.7743 | 0.0127 | 0.2500 | -5.0659 | -1.0000 |
| training_default50_by_seed | yolov8n vs yolo11n | test_detection_mAP50 | yolov8n | yolo11n | 3 | 0.8584 | 0.8686 | -0.0102 | 0.0014 | -12.4085 | 0.0064 | 0.2500 | -7.1641 | -1.0000 |
| training_minimal_aug20_by_seed | yolov8n vs yolo11n | test_count_mae | yolov8n | yolo11n | 3 | 4.7778 | 4.2741 | 0.5037 | 1.1976 | 0.7285 | 0.5421 | 0.5000 | 0.4206 | 0.6667 |
| training_minimal_aug20_by_seed | yolov8n vs yolo11n | test_count_rmse | yolov8n | yolo11n | 3 | 12.6496 | 11.9776 | 0.6720 | 5.2455 | 0.2219 | 0.8450 | 0.7500 | 0.1281 | 0.3333 |
| training_minimal_aug20_by_seed | yolov8n vs yolo11n | test_count_r2 | yolov8n | yolo11n | 3 | 0.8886 | 0.8970 | -0.0084 | 0.0920 | -0.1584 | 0.8887 | 1.0000 | -0.0914 | 0.0000 |
| training_minimal_aug20_by_seed | yolov8n vs yolo11n | test_detection_mAP50 | yolov8n | yolo11n | 3 | 0.8245 | 0.8147 | 0.0098 | 0.0066 | 2.5833 | 0.1228 | 0.2500 | 1.4914 | 1.0000 |
| training_robust_aug20_by_seed | yolov8n vs yolo11n | test_count_mae | yolov8n | yolo11n | 3 | 9.6148 | 11.7259 | -2.1111 | 2.0513 | -1.7825 | 0.2166 | 0.2500 | -1.0291 | -1.0000 |
| training_robust_aug20_by_seed | yolov8n vs yolo11n | test_count_rmse | yolov8n | yolo11n | 3 | 22.8303 | 25.3928 | -2.5625 | 4.2023 | -1.0562 | 0.4016 | 0.5000 | -0.6098 | -0.6667 |
| training_robust_aug20_by_seed | yolov8n vs yolo11n | test_count_r2 | yolov8n | yolo11n | 3 | 0.6429 | 0.5571 | 0.0858 | 0.1401 | 1.0608 | 0.3999 | 0.5000 | 0.6125 | 0.6667 |
| training_robust_aug20_by_seed | yolov8n vs yolo11n | test_detection_mAP50 | yolov8n | yolo11n | 3 | 0.8494 | 0.8587 | -0.0093 | 0.0150 | -1.0800 | 0.3931 | 0.5000 | -0.6235 | -0.6667 |
| training_case_sensitivity_yolo11n_by_seed | default50 vs default20 | test_count_mae | default50 | default20 | 3 | 9.2148 | 7.9926 | 1.2222 | 2.7035 | 0.7831 | 0.5156 | 1.0000 | 0.4521 | 0.0000 |
| training_case_sensitivity_yolo11n_by_seed | default50 vs default20 | test_detection_mAP50 | default50 | default20 | 3 | 0.8686 | 0.8479 | 0.0207 | 0.0029 | 12.3752 | 0.0065 | 0.2500 | 7.1448 | 1.0000 |
| training_case_sensitivity_yolo11n_by_seed | minimal_aug20 vs default20 | test_count_mae | minimal_aug20 | default20 | 3 | 4.2741 | 7.9926 | -3.7185 | 1.5125 | -4.2584 | 0.0510 | 0.2500 | -2.4586 | -1.0000 |
| training_case_sensitivity_yolo11n_by_seed | minimal_aug20 vs default20 | test_detection_mAP50 | minimal_aug20 | default20 | 3 | 0.8147 | 0.8479 | -0.0333 | 0.0105 | -5.4781 | 0.0317 | 0.2500 | -3.1628 | -1.0000 |
| training_case_sensitivity_yolo11n_by_seed | robust_aug20 vs default20 | test_count_mae | robust_aug20 | default20 | 3 | 11.7259 | 7.9926 | 3.7333 | 1.8860 | 3.4286 | 0.0756 | 0.2500 | 1.9795 | 1.0000 |
| training_case_sensitivity_yolo11n_by_seed | robust_aug20 vs default20 | test_detection_mAP50 | robust_aug20 | default20 | 3 | 0.8587 | 0.8479 | 0.0108 | 0.0080 | 2.3476 | 0.1434 | 0.2500 | 1.3554 | 1.0000 |
| training_case_sensitivity_yolov8n_by_seed | default50 vs default20 | test_count_mae | default50 | default20 | 3 | 14.3333 | 10.0815 | 4.2519 | 1.7798 | 4.1379 | 0.0537 | 0.2500 | 2.3890 | 1.0000 |
| training_case_sensitivity_yolov8n_by_seed | default50 vs default20 | test_detection_mAP50 | default50 | default20 | 3 | 0.8584 | 0.8600 | -0.0017 | 0.0106 | -0.2700 | 0.8125 | 1.0000 | -0.1559 | 0.0000 |
| training_case_sensitivity_yolov8n_by_seed | minimal_aug20 vs default20 | test_count_mae | minimal_aug20 | default20 | 3 | 4.7778 | 10.0815 | -5.3037 | 0.8562 | -10.7297 | 0.0086 | 0.2500 | -6.1948 | -1.0000 |
| training_case_sensitivity_yolov8n_by_seed | minimal_aug20 vs default20 | test_detection_mAP50 | minimal_aug20 | default20 | 3 | 0.8245 | 0.8600 | -0.0356 | 0.0102 | -6.0561 | 0.0262 | 0.2500 | -3.4965 | -1.0000 |
| training_case_sensitivity_yolov8n_by_seed | robust_aug20 vs default20 | test_count_mae | robust_aug20 | default20 | 3 | 9.6148 | 10.0815 | -0.4667 | 0.4494 | -1.7985 | 0.2139 | 0.2500 | -1.0384 | -1.0000 |
| training_case_sensitivity_yolov8n_by_seed | robust_aug20 vs default20 | test_detection_mAP50 | robust_aug20 | default20 | 3 | 0.8494 | 0.8600 | -0.0106 | 0.0097 | -1.8974 | 0.1982 | 0.2500 | -1.0955 | -1.0000 |

Statistical test CSV: `/workspace/reports/paper_q4_prep/statistical_tests.csv`