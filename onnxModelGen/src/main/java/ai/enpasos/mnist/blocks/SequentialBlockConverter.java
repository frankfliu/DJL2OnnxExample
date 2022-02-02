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
import ai.djl.nn.SequentialBlock;
import ai.djl.util.Pair;

import java.util.List;

import static ai.enpasos.mnist.blocks.OnnxHelper.createValueInfoProto;

public class SequentialBlockConverter implements OnnxConverter {

    @Override
    public OnnxBlock getOnnxBlock(Block block, OnnxCounter counter, List<OnnxTensor> input) {
        SequentialBlock seq = (SequentialBlock) block;
        OnnxBlock onnxBlock = OnnxBlock.builder()
                .input(input)
                .valueInfos(createValueInfoProto(input))
                .build();

        List<OnnxTensor> currentInput = input;
        for (Pair<String, Block> p : seq.getChildren()) {
            OnnxConverter converter = OnnxConverter.getConverter(p.getValue());
            OnnxBlock child = converter.getOnnxBlock(p.getValue(), counter, currentInput);
            onnxBlock.addChild(child);

            currentInput = child.getOutput();
        }

        onnxBlock.setOutput(currentInput);
        return onnxBlock;
    }
}
