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
import ai.djl.nn.norm.LayerNorm;
import ai.enpasos.onnx.AttributeProto;
import ai.enpasos.onnx.NodeProto;
import ai.enpasos.onnx.TensorProto;

import java.util.List;

import static ai.enpasos.mnist.blocks.OnnxBlock.combine;
import static ai.enpasos.mnist.blocks.OnnxHelper.convert;
import static ai.enpasos.mnist.blocks.OnnxHelper.createValueInfoProto;

public class LayerNormConvert implements OnnxConverter {

    @Override
    public OnnxBlock getOnnxBlock(Block block, OnnxCounter counter, List<OnnxTensor> input) {
        LayerNorm layerNorm = (LayerNorm) block;
        OnnxBlock blockMVN = nodeMVN(counter, input);
        OnnxBlock blockMul = nodeMul(layerNorm, counter, blockMVN.getOutput());
        OnnxBlock blockAdd = nodeAdd(layerNorm, counter, blockMul.getOutput());

        OnnxBlock onnxBlock = OnnxBlock.builder()
                .input(input)
                .output(blockAdd.getOutput())
                .valueInfos(createValueInfoProto(blockAdd.getOutput()))
                .build();

        onnxBlock.addChild(blockMVN);
        onnxBlock.addChild(blockMul);
        onnxBlock.addChild(blockAdd);

        return onnxBlock;
    }

    private OnnxBlock nodeMVN(OnnxCounter counter, List<OnnxTensor> input) {
        List<OnnxTensor> output = combine(
                List.of("T" + counter.count()),
                List.of(input.get(0).getShape())
        );
        return OnnxBlock.builder()
                .output(output)
                .valueInfos(createValueInfoProto(output))
                .nodes(List.of(
                        NodeProto.newBuilder()
                                .setName("Node" + counter.count())
                                .setOpType("MeanVarianceNormalization")
                                .addAttribute(AttributeProto.newBuilder()
                                        .setType(AttributeProto.AttributeType.INTS)
                                        .setName("axes")
                                        .addAllInts(List.of(1L, 2L, 3L))
                                        .build())
                                .addInput(input.get(0).getName())
                                .addOutput(output.get(0).getName())
                                .build()
                ))
                .valueInfos(createValueInfoProto(output))
                .build();
    }

    private OnnxBlock nodeMul(LayerNorm layerNorm, OnnxCounter counter, List<OnnxTensor> input) {
        List<OnnxTensor> output = combine(
                List.of("T" + counter.count()),
                List.of(input.get(0).getShape())
        );
        String gammaName = "gamma" + counter.count();
        NDArray gamma = layerNorm.getDirectParameters().get("gamma").getArray();
        return OnnxBlock.builder()
                .output(output)
                .valueInfos(createValueInfoProto(output))
                .nodes(List.of(
                        NodeProto.newBuilder()
                                .setName("N" + counter.count())
                                .setOpType("Mul")
                                .addInput(input.get(0).getName())
                                .addInput(gammaName)
                                .addOutput(output.get(0).getName())
                                .build()
                ))
                .parameters(List.of(
                        TensorProto.newBuilder()
                                .setName(gammaName)
                                .setDataType(1)
                                .addAllDims(convert(gamma.getShape().getShape()))
                                .addAllFloatData(convert(gamma))
                                .build()
                ))
                .valueInfos(createValueInfoProto(output))
                .build();
    }

    private OnnxBlock nodeAdd(LayerNorm layerNorm, OnnxCounter counter, List<OnnxTensor> input) {
        List<OnnxTensor> output = combine(
                List.of("T" + counter.count()),
                List.of(input.get(0).getShape())
        );
        String betaName = "beta" + counter.count();
        NDArray beta = layerNorm.getDirectParameters().get("beta").getArray();
        return OnnxBlock.builder()
                .output(output)
                .valueInfos(createValueInfoProto(output))
                .nodes(List.of(
                        NodeProto.newBuilder()
                                .setName("N" + counter.count())
                                .setOpType("Add")
                                .addInput(input.get(0).getName())
                                .addInput(betaName)
                                .addOutput(output.get(0).getName())
                                .build()
                ))
                .parameters(List.of(
                        TensorProto.newBuilder()
                                .setName(betaName)
                                .setDataType(1)
                                .addAllDims(convert(beta.getShape().getShape()))
                                .addAllFloatData(convert(beta))
                                .build()
                ))
                .valueInfos(createValueInfoProto(output))
                .build();
    }
}
