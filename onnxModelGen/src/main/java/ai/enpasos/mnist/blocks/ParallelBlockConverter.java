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
import ai.djl.nn.ParallelBlock;
import ai.djl.util.Pair;
import ai.enpasos.onnx.AttributeProto;
import ai.enpasos.onnx.NodeProto;

import java.util.ArrayList;
import java.util.List;

import static ai.enpasos.mnist.blocks.OnnxBlock.createOutput;
import static ai.enpasos.mnist.blocks.OnnxBlock.getNames;
import static ai.enpasos.mnist.blocks.OnnxHelper.createValueInfoProto;

public class ParallelBlockConverter implements OnnxConverter {

    @Override
    public OnnxBlock getOnnxBlock(Block block, OnnxCounter counter, List<OnnxTensor> input) {
        ParallelBlock parallel = (ParallelBlock) block;
        OnnxBlock onnxBlock = OnnxBlock.builder()
                .input(input)
                .valueInfos(createValueInfoProto(input))
                .build();

        List<OnnxTensor> outputsToBeConcatenated = new ArrayList<>();
        for (Pair<String, Block> p : parallel.getChildren()) {
            OnnxConverter converter = OnnxConverter.getConverter(p.getValue());
            OnnxBlock child = converter.getOnnxBlock(p.getValue(), counter, input);
            onnxBlock.addChild(child);
            if (child.getOutput().size() > 1) {
                throw new RuntimeException("each output is assumed to be a single tensor here");
            }
            outputsToBeConcatenated.add(child.getOutput().get(0));
        }

        List<OnnxTensor> output =
                createOutput(List.of("T" + counter.count()), input, parallel::getOutputShapes);

        OnnxBlock concatBlock = OnnxBlock.builder()
                .input(outputsToBeConcatenated)
                .output(output)
                .valueInfos(createValueInfoProto(output))
                .nodes(List.of(
                        NodeProto.newBuilder()
                                .setName("N" + counter.count())
                                .setOpType("Concat")
                                .addAttribute(AttributeProto.newBuilder()
                                        .setType(AttributeProto.AttributeType.INT)
                                        .setName("axis")
                                        .setI(1)
                                        .build())
                                .addAllInput(getNames(outputsToBeConcatenated))
                                .addOutput(output.get(0).getName())
                                .build()
                ))
                .valueInfos(createValueInfoProto(output))
                .build();

        onnxBlock.addChild(concatBlock);
        onnxBlock.setOutput(concatBlock.getOutput());

        return onnxBlock;
    }

}
