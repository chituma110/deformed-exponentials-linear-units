# deformed-exponentials-linear-units-DELUS-
Repository for my new paper,《PDELU: Parametric Deformable Exponentials Linear Units》Activation function is the pivotal component of deep neural networks. In this repository, we introduce the deformed exponential function  and  learnable parameters into the ELUs  to combine the advantages resulting to bring benefits to the classification performance and the convergence of deep networks

## Usage

```
// PNELU layer
optional PNELUParameter pnelu_param = 164;

// PDELU: Parametric Deformable Exponentials Linear Units
message PNELUParameter {
  optional FillerParameter filler = 1;
  // Whether or not slope parameters are shared across channels.
  optional bool channel_shared = 2 [default = false];
  // Described in:
  optional float t = 3 [default = 1.5];
  //optional float namda = 4 [default = 1.051];
}

```