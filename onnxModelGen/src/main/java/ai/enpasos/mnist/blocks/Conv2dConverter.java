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

import ai.djl.ndarray.NDArray;
import ai.djl.nn.Block;
import ai.djl.nn.convolutional.Conv2d;
import ai.enpasos.onnx.AttributeProto;
import ai.enpasos.onnx.NodeProto;
import ai.enpasos.onnx.TensorProto;
import com.google.protobuf.ByteString;

import java.util.List;

import static ai.enpasos.mnist.blocks.OnnxBlock.createOutput;
import static ai.enpasos.mnist.blocks.OnnxHelper.convert;
import static ai.enpasos.mnist.blocks.OnnxHelper.createValueInfoProto;

public class Conv2dConverter implements OnnxConverter {

    @Override
    public OnnxBlock getOnnxBlock(Block block, OnnxCounter counter, List<OnnxTensor> input) {
        Conv2d conv2d = (Conv2d) block;
        List<OnnxTensor> output =
                createOutput(List.of("T" + counter.count()), input, conv2d::getOutputShapes);
        NDArray weights = conv2d.getDirectParameters().get("weight").getArray();
        String parameterName = "P" + counter.count();

        return OnnxBlock.builder()
                .input(input)
                .output(output)
                .valueInfos(createValueInfoProto(output))
                .parameters(List.of(
                        TensorProto.newBuilder()
                                .setName(parameterName)
                                .setDataType(1)
                                .addAllDims(convert(weights.getShape().getShape()))
                                .addAllFloatData(convert(weights))
                                .build()
                ))
                .nodes(List.of(
                                NodeProto.newBuilder()
                                        .setName("N" + counter.count())
                                        .setOpType("Conv")
                                        .addAttribute(AttributeProto.newBuilder()
                                                .setType(AttributeProto.AttributeType.STRING)
                                                .setName("auto_pad")
                                                .setS(ByteString.copyFromUtf8("SAME_UPPER"))
                                                .build())
                                        .addAttribute(AttributeProto.newBuilder()
                                                .setType(AttributeProto.AttributeType.INTS)
                                                .setName("dilations")
                                                .addAllInts(convert(conv2d.getDilation().getShape()))
                                                .build())
                                        .addAttribute(AttributeProto.newBuilder()
                                                .setType(AttributeProto.AttributeType.INT)
                                                .setName("group")
                                                .setI(conv2d.getGroups())
                                                .build())
                                        .addAttribute(AttributeProto.newBuilder()
                                                .setType(AttributeProto.AttributeType.INTS)
                                                .setName("kernel_shape")
                                                .addAllInts(convert(conv2d.getKernelShape().getShape()))
                                                .build())
                                        .addAttribute(AttributeProto.newBuilder()
                                                .setType(AttributeProto.AttributeType.INTS)
                                                .setName("strides")
                                                .addAllInts(convert(conv2d.getStride().getShape()))
                                                .build())
                                        .addInput(input.get(0).getName())
                                        .addInput(parameterName)
                                        .addOutput(output.get(0).getName())
                                        .build()
                        )
                ).build();
    }

}
