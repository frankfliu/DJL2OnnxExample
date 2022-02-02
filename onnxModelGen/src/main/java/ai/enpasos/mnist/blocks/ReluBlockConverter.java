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
import ai.enpasos.onnx.NodeProto;

import java.util.List;

import static ai.enpasos.mnist.blocks.OnnxBlock.combine;
import static ai.enpasos.mnist.blocks.OnnxHelper.createValueInfoProto;

public class ReluBlockConverter implements OnnxConverter {

    @Override
    public OnnxBlock getOnnxBlock(Block block, OnnxCounter counter, List<OnnxTensor> input) {
        List<OnnxTensor> output = combine(
                List.of("T" + counter.count()),
                List.of(input.get(0).getShape())
        );
        return OnnxBlock.builder()
                .input(input)
                .output(output)
                .valueInfos(createValueInfoProto(output))
                .nodes(List.of(
                        NodeProto.newBuilder()
                                .setName("reluNode" + counter.count())
                                .setOpType("Relu")
                                .addInput(input.get(0).getName())
                                .addOutput(output.get(0).getName())
                                .build()
                ))
                .valueInfos(createValueInfoProto(output))
                .build();
    }
}
