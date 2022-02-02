/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.enpasos.mnist.blocks;

import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.norm.LayerNorm;

import java.util.List;

public interface OnnxConverter {

    OnnxBlock getOnnxBlock(Block block, OnnxCounter counter, List<OnnxTensor> input);

    static OnnxConverter getConverter(Block block) {
        if (block instanceof SequentialBlock) {
            return new SequentialBlockConverter();
        } else if (block instanceof ParallelBlock) {
            return new ParallelBlockConverter();
        } else if (block instanceof LinearConverter) {
            return new LayerNormConvert();
        } else if (block instanceof LayerNorm) {
            return new LayerNormConvert();
        } else if (block instanceof Conv2d) {
            return new Conv2dConverter();
        } else if (block instanceof LambdaBlock) {
            return new LambdaBlockConverter();
        }
        throw new UnsupportedOperationException("Unsupported block type: " + block.getClass().getName());
    }
}
