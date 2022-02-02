/*
 *  Copyright (c) 2021 enpasos GmbH
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */
package ai.enpasos.mnist.blocks;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.nn.pooling.Pool;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public final class MnistBlock {

    private MnistBlock() {
    }

    public static SequentialBlock newMnistBlock() {
        return new SequentialBlock()
                .add(Conv2d.builder()
                        .setFilters(8)
                        .setKernelShape(new Shape(5, 5))
                        .optBias(false)
                        .optPadding(new Shape(2, 2))
                        .build())
                .add(LayerNorm.builder().build())
                .add(Activation.reluBlock())
                .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)))   // 28 -> 14
                .add(
                        new ParallelBlock(
                                list -> {
                                    List<NDArray> concatenatedList =
                                            list.stream().map(NDList::head).collect(Collectors.toList());
                                    return new NDList(NDArrays.concat(new NDList(concatenatedList), 1));
                                },
                                Arrays.asList(
                                        Conv2d.builder()
                                                .setFilters(16)
                                                .setKernelShape(new Shape(5, 5))
                                                .optBias(false)
                                                .optPadding(new Shape(2, 2))
                                                .build(),
                                        Conv2d.builder()
                                                .setFilters(16)
                                                .setKernelShape(new Shape(3, 3))
                                                .optBias(false)
                                                .optPadding(new Shape(1, 1))
                                                .build()
                                ))
                )
                .add(LayerNorm.builder().build())
                .add(Activation.reluBlock())
                .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)))  // 14 -> 7
                .add(Conv2d.builder()
                        .setFilters(32)
                        .setKernelShape(new Shape(3, 3))
                        .optBias(false)
                        .optPadding(new Shape(1, 1))
                        .build())
                .add(LayerNorm.builder().build())
                .add(Activation.reluBlock())
                .add(inputs -> {
                    NDArray current = inputs.head();

                    // Scale to the range [0, 1]  (same range as the action input)
                    Shape origShape = current.getShape();
                    Shape shape2 =
                            new Shape(origShape.get(0), origShape.get(1) * origShape.get(2) * origShape.get(3));
                    NDArray current2 = current.reshape(shape2);
                    Shape shape3 = new Shape(origShape.get(0), 1, 1, 1);
                    NDArray min2 = current2.min(new int[]{1}, true).reshape(shape3);
                    NDArray max2 = current2.max(new int[]{1}, true).reshape(shape3);

                    NDArray d = max2.sub(min2).maximum(1e-5);

                    NDArray a = current.sub(min2);
                    return new NDList(a.div(d));
                }, "rescale")
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder()
                        .setUnits(10)
                        .optBias(true)
                        .build());
    }
}
