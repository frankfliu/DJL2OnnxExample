package ai.enpasos.mnist.blocks;

import ai.djl.nn.Block;

import java.util.List;

public interface OnnxIO {
    OnnxBlock getOnnxBlock(Block block, OnnxCounter counter, List<OnnxTensor> input);
}
