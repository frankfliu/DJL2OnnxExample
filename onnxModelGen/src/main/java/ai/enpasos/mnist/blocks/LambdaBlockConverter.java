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
import ai.enpasos.onnx.AttributeProto;
import ai.enpasos.onnx.NodeProto;
import ai.enpasos.onnx.TensorProto;
import com.google.protobuf.ByteString;
import org.apache.commons.lang3.NotImplementedException;

import java.util.List;

import static ai.enpasos.mnist.blocks.OnnxBlock.createOutput;
import static ai.enpasos.mnist.blocks.OnnxHelper.createValueInfoProto;

public class LambdaBlockConverter implements OnnxConverter {

    @Override
    public OnnxBlock getOnnxBlock(Block block, OnnxCounter ctx, List<OnnxTensor> input) {
        LambdaBlock lambda = (LambdaBlock) block;
        if ("rescale".equals(lambda.getName())) {
            return new RescaleBlockConverter().getOnnxBlock(block, ctx, input);
        }

        String outputName = "T" + ctx.count();
        List<OnnxTensor> output = createOutput(List.of(outputName), input, lambda::getOutputShapes);
        OnnxBlock onnxBlock = OnnxBlock.builder()
                .input(input)
                .output(output)
                .valueInfos(createValueInfoProto(output))
                .build();

        NodeProto.Builder nodeBuilder = NodeProto.newBuilder()
                .addInput(input.get(0).getName())
                .addOutput(outputName)
                .setName("N" + ctx.count());

        String name = lambda.getName();
        switch (name) {
            case "maxPool2d":
                nodeBuilder
                        .setOpType("MaxPool")
                        .addAttribute(AttributeProto.newBuilder()
                                .setType(AttributeProto.AttributeType.STRING)
                                .setName("auto_pad")
                                .setS(ByteString.copyFromUtf8("NOTSET"))
                                .build())
                        .addAttribute(AttributeProto.newBuilder()
                                .setType(AttributeProto.AttributeType.INTS)
                                .setName("kernel_shape")
                                .addAllInts(List.of(2L, 2L))
                                .build())
                        .addAttribute(AttributeProto.newBuilder()
                                .setType(AttributeProto.AttributeType.INTS)
                                .setName("strides")
                                .addAllInts(List.of(2L, 2L))
                                .build());

                break;
            case "batchFlatten":
                String shapeName = "batchFlattenNodeShape" + ctx.count();
                nodeBuilder.setOpType("Reshape")
                        .addInput(shapeName);
                long size = input.get(0).getShape().size();
                onnxBlock.getParameters().add(TensorProto.newBuilder()
                        .setName(shapeName)
                        .setDataType(TensorProto.INT64_DATA_FIELD_NUMBER)
                        .addAllDims(List.of(2L))
                        .addAllInt64Data(List.of(1L, size))
                        .build());
                break;
            case "Relu":
                nodeBuilder.setOpType("Relu");
                break;
            case "identity":
            case "anonymous":
            default:
                throw new NotImplementedException(name);
        }
        onnxBlock.getNodes().add(nodeBuilder.build());

        return onnxBlock;
    }
}
