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

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.enpasos.onnx.AttributeProto;
import ai.enpasos.onnx.NodeProto;

import java.util.List;

import static ai.enpasos.mnist.blocks.OnnxBlock.combine;
import static ai.enpasos.mnist.blocks.OnnxHelper.createValueInfoProto;

public class RescaleBlockConverter implements OnnxConverter {

    @Override
    public OnnxBlock getOnnxBlock(Block block, OnnxCounter counter, List<OnnxTensor> input) {
        OnnxBlock blockMin = nodeMin(counter, input);
        OnnxBlock blockMax = nodeMax(counter, input);
        OnnxBlock blockSubA = nodeSub(counter, input, blockMin.getOutput());
        OnnxBlock blockSubB = nodeSub(counter, blockMax.getOutput(), blockMin.getOutput());
        OnnxBlock blockDiv = nodeDiv(counter, blockSubA.getOutput(), blockSubB.getOutput());

        OnnxBlock onnxBlock = OnnxBlock.builder()
                .input(input)
                .output(blockDiv.getOutput())
                .valueInfos(createValueInfoProto(blockDiv.getOutput()))
                .build();

        onnxBlock.addChild(blockMin);
        onnxBlock.addChild(blockMax);
        onnxBlock.addChild(blockSubA);
        onnxBlock.addChild(blockSubB);
        onnxBlock.addChild(blockDiv);

        return onnxBlock;

    }

    private OnnxBlock nodeMin(OnnxCounter counter, List<OnnxTensor> input) {
        List<OnnxTensor> output = combine(
                List.of("T" + counter.count()),
                List.of(new Shape(input.get(0).getShape().get(0), 1, 1, 1))
        );
        return OnnxBlock.builder()
                .output(output)
                .valueInfos(createValueInfoProto(output))
                .nodes(List.of(
                        NodeProto.newBuilder()
                                .setName("N" + counter.count())
                                .setOpType("ReduceMin")
                                .addAttribute(AttributeProto.newBuilder()
                                        .setType(AttributeProto.AttributeType.INTS)
                                        .setName("axes")
                                        .addAllInts(List.of(1L, 2L, 3L))
                                        .build())
                                .addInput(input.get(0).getName())
                                .addOutput(output.get(0).getName())
                                .build())
                )
                .build();
    }

    private OnnxBlock nodeMax(OnnxCounter counter, List<OnnxTensor> input) {
        List<OnnxTensor> output = combine(
                List.of("T" + counter.count()),
                List.of(new Shape(input.get(0).getShape().get(0), 1, 1, 1))
        );
        return OnnxBlock.builder()
                .output(output)
                .valueInfos(createValueInfoProto(output))
                .nodes(List.of(
                        NodeProto.newBuilder()
                                .setName("N" + counter.count())
                                .setOpType("ReduceMax")
                                .addAttribute(AttributeProto.newBuilder()
                                        .setType(AttributeProto.AttributeType.INTS)
                                        .setName("axes")
                                        .addAllInts(List.of(1L, 2L, 3L))
                                        .build())
                                .addInput(input.get(0).getName())
                                .addOutput(output.get(0).getName())
                                .build())
                )
                .build();
    }

    private OnnxBlock nodeSub(OnnxCounter counter, List<OnnxTensor> inputA, List<OnnxTensor> inputB) {
        List<OnnxTensor> output = combine(
                List.of("T" + counter.count()),
                List.of(inputA.get(0).getShape())
        );
        return OnnxBlock.builder()
                .output(output)
                .valueInfos(createValueInfoProto(output))
                .nodes(List.of(
                        NodeProto.newBuilder()
                                .setName("N" + counter.count())
                                .setOpType("Sub")
                                .addInput(inputA.get(0).getName())
                                .addInput(inputB.get(0).getName())
                                .addOutput(output.get(0).getName())
                                .build()
                ))
                .valueInfos(createValueInfoProto(output))
                .build();
    }

    private OnnxBlock nodeDiv(OnnxCounter counter, List<OnnxTensor> inputA, List<OnnxTensor> inputB) {
        List<OnnxTensor> output = combine(
                List.of("T" + counter.count()),
                List.of(inputA.get(0).getShape())
        );
        return OnnxBlock.builder()
                .output(output)
                .valueInfos(createValueInfoProto(output))
                .nodes(List.of(
                        NodeProto.newBuilder()
                                .setName("N" + counter.count())
                                .setOpType("Div")
                                .addInput(inputA.get(0).getName())
                                .addInput(inputB.get(0).getName())
                                .addOutput(output.get(0).getName())
                                .build()
                ))
                .valueInfos(createValueInfoProto(output))
                .build();
    }
}
