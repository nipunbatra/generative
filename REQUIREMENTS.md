# Requirements to scene map

| Requirement | Scenes | Evidence |
|---|---|---|
| 1. Scene inventory | s01–s60 | [src/scene-manifest.json](src/scene-manifest.json) |
| 2. 120/90-minute routes | all | [src/content.mjs](src/content.mjs) |
| 3. Ten application opening | s01–s10 | [assets/manifest.json](assets/manifest.json) |
| 4. Seven ELBO steps | s27–s28 | [src/content.mjs](src/content.mjs) |
| 5. KL arithmetic | s19–s22 | [qa/math-checks.json](qa/math-checks.json) |
| 6. KL support | s21 | [qa/math-checks.json](qa/math-checks.json) |
| 7. Finite ELBO calculator | s30 | [qa/math-checks.json](qa/math-checks.json) |
| 8. Two different KLs | s29–s31 | [src/content.mjs](src/content.mjs) |
| 9. Pathwise derivative | s32–s36 | [qa/lab-checks.json](qa/lab-checks.json) |
| 10. Two-pixel loss | s40 | [qa/lab-checks.json](qa/lab-checks.json) |
| 11. Tensor shapes | s37,s46,s53 | [qa/lab-checks.json](qa/lab-checks.json) |
| 12. Likelihood and reductions | s38–s41 | [qa/lab-checks.json](qa/lab-checks.json) |
| 13. Prior generation | s42–s43 | [qa/lab-checks.json](qa/lab-checks.json) |
| 14. GAN objectives | s47–s51 | [qa/math-checks.json](qa/math-checks.json) |
| 15. Gradient isolation | s48–s52 | [qa/lab-checks.json](qa/lab-checks.json) |
| 16. Measured monitoring and coverage | s55–s56 | [assets/recorded/provenance.json](assets/recorded/provenance.json) |
| 17. Interaction and reset | all demos | [qa/browser-checks.json](qa/browser-checks.json) |
| 18. Visual/export QA | all | [qa/QA_REPORT.md](qa/QA_REPORT.md) |
| 19. Offline provenance | all | [assets/manifest.json](assets/manifest.json) |
| 20. Complete package | all | [README.md](README.md) |

The baseline core count is unchanged: 60 scenes, 145 meaningful reveal states. Three optional appendix scenes add six builds. The completed PDF adds one seven-step ELBO continuation page. The appendix and continuation are outside both route timings.
