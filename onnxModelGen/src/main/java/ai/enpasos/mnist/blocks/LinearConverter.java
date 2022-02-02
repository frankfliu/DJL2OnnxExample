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
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.core.Linear;
import ai.enpasos.onnx.NodeProto;
import ai.enpasos.onnx.TensorProto;

import java.util.List;

import static ai.enpasos.mnist.blocks.OnnxBlock.combine;
import static ai.enpasos.mnist.blocks.OnnxBlock.createOutput;
import static ai.enpasos.mnist.blocks.OnnxHelper.convert;
import static ai.enpasos.mnist.blocks.OnnxHelper.createValueInfoProto;

public class LinearConverter implements OnnxConverter {

    @Override
    public OnnxBlock getOnnxBlock(Block block, OnnxCounter counter, List<OnnxTensor> input) {
        Linear linear = (Linear) block;

        List<OnnxTensor> output =
                createOutput(List.of("T" + counter.count()), input, linear::getOutputShapes);

        Shape outputShape = output.get(0).getShape();

        long inputDim = input.get(0).getShape().get(1);
        long outputDim = outputShape.get(1);

        OnnxBlock blockW = nodeW(linear, counter, new Shape(inputDim, outputDim));
        OnnxBlock blockMult = nodeMult(counter, input.get(0), blockW.getOutput().get(0));
        OnnxBlock blockB = nodeB(linear, counter, blockMult.getOutput());

        OnnxBlock onnxBlock = OnnxBlock.builder()
                .input(input)
                .output(blockB.getOutput())
                .build();

        onnxBlock.addChild(blockW);
        onnxBlock.addChild(blockMult);
        onnxBlock.addChild(blockB);

        return onnxBlock;
    }

    private OnnxBlock nodeB(Linear linear, OnnxCounter counter, List<OnnxTensor> input) {
        List<OnnxTensor> output = combine(
                List.of("T" + counter.count()),
                List.of(input.get(0).getShape())
        );
        String parameterName = "P" + counter.count();
        return OnnxBlock.builder()
                .output(output)
                .valueInfos(createValueInfoProto(output))
                .nodes(List.of(
                        NodeProto.newBuilder()
                                .setName("N" + counter.count())
                                .setOpType("Add")
                                .addInput(input.get(0).getName())
                                .addInput(parameterName)
                                .addOutput(output.get(0).getName())
                                .build()
                ))
                .parameters(List.of(
                        TensorProto.newBuilder()
                                .setName(parameterName)
                                .setDataType(1)
                                .addAllDims(List.of(1L, 10L))
                                .addAllFloatData(convert(linear.getParameters().get("bias").getArray()))
                                .build()
                ))
                .build();
    }

    private OnnxBlock nodeMult(OnnxCounter counter, OnnxTensor inputA, OnnxTensor inputB) {
        List<OnnxTensor> output = combine(
                List.of("T" + counter.count()),
                List.of(new Shape(inputA.getShape().get(0), inputB.getShape().get(1)))
        );
        return OnnxBlock.builder()
                .output(output)
                .valueInfos(createValueInfoProto(output))
                .nodes(List.of(
                        NodeProto.newBuilder()
                                .setName("Mul" + counter.count())
                                .setOpType("MatMul")
                                .addInput(inputA.getName())
                                .addInput(inputB.getName())
                                .addOutput(output.get(0).getName())
                                .build()
                ))
                .build();
    }

    private OnnxBlock nodeW(Linear linear, OnnxCounter ctx, Shape outputShape) {
        List<OnnxTensor> output = combine(List.of("T" + ctx.count()), List.of(outputShape));

        NDArray weight = linear.getDirectParameters().get("weight").getArray().transpose();

        String parameterName1 = "P" + ctx.count();
        String parameterName2 = "P" + ctx.count();

        return OnnxBlock.builder()
                .output(output)
                .valueInfos(createValueInfoProto(output))
                .nodes(List.of(
                        NodeProto.newBuilder()
                                .setName("Node" + ctx.count())
                                .setOpType("Reshape")
                                .addInput(parameterName1)
                                .addInput(parameterName2)
                                .addOutput(output.get(0).getName())
                                .build()
                ))
                .parameters(List.of(
                        // data
                        TensorProto.newBuilder()
                                .setName(parameterName1)
                                .setDataType(1)
                                .addAllDims(convert(weight.getShape().getShape()))
                                .addAllFloatData(convert(weight))
                                .build(),
                        // shape
                        TensorProto.newBuilder()
                                .setName(parameterName2)
                                .setDataType(TensorProto.INT64_DATA_FIELD_NUMBER)
                                .addAllDims(List.of(2L))
                                .addAllInt64Data(convert(outputShape.getShape()))
                                .build()
                ))
                .valueInfos(
                        createValueInfoProto(output)
                )
                .build();

    }
}
