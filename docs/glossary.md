# Glossary

Terms commonly used in *FastAI.jl*.



- **data container**

    A data structure that is used to load a number of data observations separately and lazily. It defines how many observations it holds with `nobs` and how to load a single observation with `getobs`. See [here](data_containers.md).