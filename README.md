# Encoder-Decoder Architecture: Step-wise Overview

## Encoder Steps

| Step | Operation                                                                                   | Shape             |
|------|---------------------------------------------------------------------------------------------|-------------------|
| Input | Input Image                                                                                | (H, W, C)         |
| Step 1 | Conv(16 filters, $3 \times 3$, LeakyReLU)                                                  | (H, W, 16)        |
| Step 2 | Conv(16 filters, $3 \times 3$, LeakyReLU)                                                  | (H, W, 16)        |
| Step 3 | MaxPool($2 \times 2$)                                                                       | (H/2, W/2, 16)    |
| Step 4 | Conv(32 filters, $3 \times 3$, LeakyReLU)                                                  | (H/2, W/2, 32)    |
| Step 5 | Conv(32 filters, $3 \times 3$, LeakyReLU)                                                  | (H/2, W/2, 32)    |
| Step 6 | MaxPool($2 \times 2$)                                                                       | (H/4, W/4, 32)    |
| Step 7 | Conv(64 filters, $3 \times 3$, LeakyReLU)                                                  | (H/4, W/4, 64)    |
| Step 8 | Conv(64 filters, $3 \times 3$, LeakyReLU)                                                  | (H/4, W/4, 64)    |
| Step 9 | MaxPool($2 \times 2$)                                                                       | (H/8, W/8, 64)    |
| Step 10 | Conv(128 filters, $3 \times 3$, LeakyReLU)                                                 | (H/8, W/8, 128)   |
| Step 11 | Conv(128 filters, $3 \times 3$, LeakyReLU)                                                 | (H/8, W/8, 128)   |
| Step 12 | MaxPool($2 \times 2$)                                                                      | (H/16, W/16, 128) |
| Step 13 | Conv(256 filters, $3 \times 3$, LeakyReLU)                                                 | (H/16, W/16, 256) |
| Step 14 | Conv(256 filters, $3 \times 3$, LeakyReLU)                                                 | (H/16, W/16, 256) |
| Step 15 | MaxPool($2 \times 2$)                                                                      | (H/32, W/32, 256) |
| Step 16 | MaxPool($2 \times 2$)                                                                      | (H/64, W/64, 256) |
| Step 17 | Compute $\mu$, $\log(\sigma^2)$                                                            | Scalar            |
| Step 18 | Parameterization                                                                          | (H/64, W/64, 256) |

## Decoder Steps

| Step | Operation                                                                                   | Shape             |
|------|---------------------------------------------------------------------------------------------|-------------------|
| Step 1 | ConvTranspose(256, $3 \times 3$)                                                             | (H/32, W/32, 256) |
| Step 2 | Add Skip Connection                                                                         | (H/32, W/32, 256) |
| Step 3 | Conv(256 filters, $3 \times 3$)                                                              | (H/32, W/32, 256) |
| Step 4 | Conv(256 filters, $3 \times 3$)                                                              | (H/32, W/32, 256) |
| Step 5 | ConvTranspose(128, $3 \times 3$)                                                             | (H/16, W/16, 128) |
| Step 6 | Add Skip Connection                                                                         | (H/16, W/16, 128) |
| Step 7 | Conv(128 filters, $3 \times 3$)                                                              | (H/16, W/16, 128) |
| Step 8 | Conv(128 filters, $3 \times 3$)                                                              | (H/16, W/16, 128) |
| Step 9 | ConvTranspose(64, $3 \times 3$)                                                              | (H/8, W/8, 64)    |
| Step 10 | Add Skip Connection                                                                         | (H/8, W/8, 64)    |
| Step 11 | Conv(64 filters, $3 \times 3$)                                                               | (H/8, W/8, 64)    |
| Step 12 | Conv(64 filters, $3 \times 3$)                                                               | (H/8, W/8, 64)    |
| Step 13 | ConvTranspose(32, $3 \times 3$)                                                              | (H/4, W/4, 32)    |
| Step 14 | Add Skip Connection                                                                         | (H/4, W/4, 32)    |
| Step 15 | Conv(32 filters, $3 \times 3$)                                                               | (H/4, W/4, 32)    |
| Step 16 | Conv(32 filters, $3 \times 3$)                                                               | (H/4, W/4, 32)    |
| Step 17 | ConvTranspose(16, $3 \times 3$)                                                              | (H/2, W/2, 16)    |
| Step 18 | Add Skip Connection                                                                         | (H/2, W/2, 16)    |
| Step 19 | Conv(16 filters, $3 \times 3$)                                                               | (H/2, W/2, 16)    |
| Step 20 | Conv(16 filters, $3 \times 3$)                                                               | (H/2, W/2, 16)    |
| Step 21 | ConvTranspose(Output, $3 \times 3$)                                                           | (H, W, Output)    |
| Step 22 | Add Skip Connection                                                                         | (H, W, Output)    |
| Step 23 | Conv(Output filters, $3 \times 3$)                                                            | (H, W, Output)    |
| Step 24 | Conv(Output filters, $3 \times 3$)                                                            | (H, W, Output)    |
