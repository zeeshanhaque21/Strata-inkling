| experiment | ti | v100-on |
|---|---|---|
| batched 1 | 67.3 (14.86 ms/step) | 64.6 (15.47 ms/step) |
| batched 2 | 107.1 (18.67 ms/step) | 109.1 (18.32 ms/step) |
| batched 3 | 136.1 (22.03 ms/step) | 145.6 (20.60 ms/step) |
| batched 4 | 159.7 (25.05 ms/step) | 174.9 (22.87 ms/step) |
| batched 8 | 234.1 (34.17 ms/step) | 269.6 (29.67 ms/step) |
| plain decode | 83.1 | 86.2 |
| prefill 20k | 1511.6 (chunk 1024) | 1575.4 (chunk 1024) |
| spec dflash2 00-py | 101.7 (acc 2.80) | 104.7 (acc 2.83) |
| spec dspark 00-py | 106.3 (acc 3.25) | 105.8 (acc 2.80) |
| spec dflash2 02-py | 146 (acc 4.00) | 161.4 (acc 4.16) |
| spec dspark 02-py | 131.2 (acc 3.55) | 138.9 (acc 3.45) |
| spec dflash2 04-cuda | 152.9 (acc 4.26) | 124.7 (acc 3.42) |
| spec dspark 04-cuda | 107 (acc 3.23) | 113.7 (acc 2.91) |
| spec dflash2 06-cuda | 88 (acc 1.94) | 87.7 (acc 1.70) |
| spec dspark 06-cuda | 94.7 (acc 2.57) | 103.5 (acc 2.59) |
| spec dflash2 08-md | 114.4 (acc 3.72) | 126.1 (acc 3.32) |
| spec dspark 08-md | 122.9 (acc 4.40) | 113.8 (acc 3.16) |
| spec dflash2 10-md | 114.8 (acc 3.89) | 92.6 (acc 2.09) |
| spec dspark 10-md | 100.2 (acc 2.72) | 81.5 (acc 1.00) |
| spec dflash2 13-cfg | 208.5 (acc 6.37) | 133.6 (acc 3.64) |
| spec dspark 13-cfg | 199 (acc 6.21) | 160.1 (acc 4.21) |
| spec dflash2 15-cfg | 211.2 (acc 6.58) | 222.8 (acc 6.54) |
| spec dspark 15-cfg | 194.7 (acc 6.00) | 201 (acc 5.76) |
| spec dflash2 chat-code | 219.8 (acc 6.20) | 232.6 (acc 6.20) |
| spec dspark chat-code | 212.4 (acc 6.02) | 225.9 (acc 6.02) |
| spec dflash2 chat-qa | 131 (acc 3.48) | 100.9 (acc 2.81) |
| spec dspark chat-qa | 79.3 (acc 0.86) | 85.8 (acc 1.28) |
| spec dflash2 code | 91.3 (acc 2.47) | 106 (acc 3.02) |
| spec dspark code | 80.4 (acc 1.69) | 136 (acc 3.91) |
| spec dflash2 code700 | 160.6 (acc 5.31) | 167.6 (acc 5.12) |
| spec dspark code700 | 86.5 (acc 2.15) | 91.7 (acc 2.15) |
| spec dflash2 segA | 82.6 (acc 1.74) | 141.2 (acc 4.68) |
| spec dspark segA | 92.9 (acc 2.30) | 84.5 (acc 1.78) |
| phase dflash2 draft | 2.572 (ms) | 2.202 (ms) |
| phase dflash2 plain decode step | 11.838 (ms) | 11.305 (ms) |
| phase dflash2 verify 8 tokens | 30.862 (ms) | 28.79 (ms) |
| phase dspark draft | 3.175 (ms) | 2.449 (ms) |
| phase dspark plain decode step | 11.818 (ms) | 11.301 (ms) |
| phase dspark verify 8 tokens | 31.243 (ms) | 29.104 (ms) |
