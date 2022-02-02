package ai.enpasos.mnist.blocks;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class OnnxCounter {
    int counter;

    public int count() {
        return counter++;
    }
}
