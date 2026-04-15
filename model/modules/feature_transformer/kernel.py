import cupy as cp
import torch


def _find_nearest_divisor(value: int, target: int) -> int:
    divisors = []
    for i in range(1, value + 1):
        if value % i == 0:
            divisors.append((i, abs(target - i)))
    divisors.sort(key=lambda x: x[1])
    return divisors[0][0]


_num_threads_forward_cache: dict[int, int] = dict()


def _get_num_threads_for_forward(output_size: int) -> int:
    optimal_num_threads = 512
    if output_size not in _num_threads_forward_cache:
        _num_threads_forward_cache[output_size] = _find_nearest_divisor(
            output_size, optimal_num_threads
        )

    return _num_threads_forward_cache[output_size]


_num_threads_backward_cache: dict[int, int] = dict()


def _get_num_threads_for_backward(output_size: int) -> int:
    optimal_num_threads = 512
    if output_size not in _num_threads_backward_cache:
        _num_threads_backward_cache[output_size] = _find_nearest_divisor(
            output_size, optimal_num_threads
        )

    return _num_threads_backward_cache[output_size]


def _kernel_with_threads(kernel, threads):
    def f(grid, args):
        kernel(grid=grid, block=threads, args=args)

    return f


_sparse_input_linear_forward_kernel_cache = dict()


@torch.compiler.disable(recursive=False)
def make_sparse_input_linear_forward_kernel(max_active_indices: int, output_size: int, dtype: torch.dtype = torch.float32):
    """
    @param: max_active_indices
        The maximum number of indices that are non-zero
        for a single position. This value determines
        the shape of the inputs.
        This value is of type uint32_t.

    @param: output_size
        The number of outputs. Must match the shape of weights
        and biases.
        This value is of type uint32.

    @param: dtype
        The data type for computations (torch.float32 or torch.bfloat16).
    """
    num_threads = _get_num_threads_for_forward(output_size)
    output_thread_slice_size = output_size // num_threads
    key = (max_active_indices, output_size, num_threads, dtype)

    use_bf16 = dtype == torch.bfloat16

    if key not in _sparse_input_linear_forward_kernel_cache:
        if use_bf16:
            data_type = "nv_bfloat16"
            compute_type = "float"
        else:
            data_type = "float"
            compute_type = "float"

        kernel = cp.RawKernel(
            r"""

typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__

/*
    @assumptions:
        The blocks must have dimensionality (BATCH_SIZE,)
        The threads must have dimensionality (N,), where
        N * output_thread_slice_size == output_size.

    @param: input_indices
        A matrix of shape (BATCH_SIZE, max_active_indices)
        containing indices of active indices for each position
        in a batch. Input index of -1 means that the slot is empty
        and the weights will not be accumulated for it. Moreover
        no further indices from this block will be considered.
        The indices form an implicit matrix of shape
        (BATCH_SIZE, NUM_INPUTS), where the first dimension index is
        inferred from the memory location (BATCH_SIZE), and the
        second dimension index is stored in the input_indices matrix.
        The type for input indices is int32_t.

    @param: input_values
        A matrix of shape (BATCH_SIZE, max_active_indices)
        containing the values (arity) of the corresponding
        input index in input_indices.
        The type for the input value (arity) is {data_type}.

    @param: weight
        The weight matrix of shape (NUM_INPUTS, output_size).
        Weights must be of type {data_type}.

    @param: bias
        The bias vector of shape (output_size,).
        Bias values must be of type {data_type}.

    @param: output
        An output matrix of shape (BATCH_SIZE, output_size).
        It may not be initialized, bias is always copied
        to the output first.
        Output values must have type {data_type}.
*/
void sparse_input_linear_forward(
    const int32_t* const input_indices,
    const {data_type}*   const input_values,
    const {data_type}*   const weight,
    const {data_type}*   const bias,
          {data_type}*   const output
) {{
    __shared__
          {compute_type}    shared_output[{output_size}];

    const uint32_t       block_idx           = blockIdx.x;
    const uint32_t       slice_offset        = threadIdx.x * {output_thread_slice_size};

          {data_type}*   const output_slice        = output + block_idx * {output_size} + slice_offset;
    const {data_type}*   const bias_slice          = bias                               + slice_offset;
          {compute_type}*   shared_output_slice = shared_output                      + slice_offset;

    const int32_t* const input_index_row     = input_indices + block_idx * {max_active_indices};
    const {data_type}*   const input_value_row     = input_values  + block_idx * {max_active_indices};

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        shared_output_slice[s] = (({compute_type})bias_slice[s]);
    }}

    for (uint32_t k = 0; k < {max_active_indices}; ++k)
    {{
        const int32_t input_index = input_index_row[k];
        const {data_type}   input_value = input_value_row[k];
        if (input_index != -1)
        {{
            const {data_type}* const weight_slice = weight + input_index * {output_size} + slice_offset;
            #pragma unroll
            for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
            {{
                shared_output_slice[s] += (({compute_type})weight_slice[s]) * (({compute_type})input_value);
            }}
        }} else break;
    }}

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        output_slice[s] = ({data_type})shared_output_slice[s];
    }}
}}

""".format(
                max_active_indices=max_active_indices,
                output_thread_slice_size=output_thread_slice_size,
                output_size=output_size,
                data_type=data_type,
                compute_type=compute_type,
            ),
            "sparse_input_linear_forward",
        )
        kernel.compile()
        _sparse_input_linear_forward_kernel_cache[key] = _kernel_with_threads(
            kernel, (num_threads,)
        )
    return _sparse_input_linear_forward_kernel_cache[key]


_sparse_input_linear_backward_kernel_cache = dict()


@torch.compiler.disable(recursive=False)
def make_sparse_input_linear_backward_kernel(max_active_indices: int, output_size: int, dtype: torch.dtype = torch.float32):
    """
    @param: max_active_indices
        The maximum number of indices that are non-zero for
        a single position. This value determines the shape
        of the inputs.
        This value is of type uint32_t.

    @param: output_size
        The number of outputs. Must match the shape of weights
        and biases.
        This value is of type uint32.

    @param: dtype
        The data type for computations (torch.float32 or torch.bfloat16).
    """
    num_threads = _get_num_threads_for_backward(output_size)
    output_thread_slice_size = output_size // num_threads
    key = (max_active_indices, output_size, num_threads, dtype)

    use_bf16 = dtype == torch.bfloat16

    if key not in _sparse_input_linear_backward_kernel_cache:
        if use_bf16:
            data_type = "nv_bfloat16"
            compute_type = "float"
        else:
            data_type = "float"
            compute_type = "float"

        kernel = cp.RawKernel(
            r"""

typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__
/*
    @assumptions:
        The blocks must have dimensionality (BATCH_SIZE,)
        The threads must have dimensionality (N,), where
        N * output_thread_slice_size == output_size.

    @param: input_indices
        A matrix of shape (BATCH_SIZE, max_active_indices)
        containing indices of active indices for each position
        in a batch. Input index of -1 means that the slot is empty
        and the weights will not be accumulated for it. Moreover
        no further indices from this block will be considered.
        The indices form an implicit matrix of shape
        (BATCH_SIZE, NUM_INPUTS), where the first dimension index is
        inferred from the memory location (BATCH_SIZE), and the
        second dimension index is stored in the input_indices matrix.
        The type for input indices is int32_t.

    @param: input_values
        A matrix of shape (BATCH_SIZE, max_active_indices)
        containing the values (arity) of the corresponding
        input index in input_indices.
        The type for the input value (arity) is {data_type}.

    @param: weight_grad
        The weight gradient matrix of shape (NUM_INPUTS, output_size).
        The gradient is accumulated, i.e. it must be zero initialized
        on the first call.
        Weights must be of type {data_type}.

    @param: bias_grad
        The bias gradient vector of shape (output_size,).
        The gradient is accumulated, i.e. it must be zero initialized
        on the first call.
        Bias values must be of type {data_type}.

    @param: output_grad
        An output gradient matrix of shape (BATCH_SIZE, output_size).
        Output values must have type {data_type}.
*/
void sparse_input_linear_backward(
    const int32_t* const input_indices,
    const {data_type}*   const input_values,
          {data_type}*   const weight_grad,
          {data_type}*   const bias_grad,
    const {data_type}*   const output_grad
) {{
    __shared__
          {compute_type}    shared_output_grad[{output_size}];

    const uint32_t       block_idx                = blockIdx.x;
    const uint32_t       slice_offset             = threadIdx.x * {output_thread_slice_size};

    const {data_type}*   const output_grad_slice        = output_grad + block_idx * {output_size} + slice_offset;
          {data_type}*   const bias_grad_slice          = bias_grad                               + slice_offset;
          {compute_type}*   shared_output_grad_slice = shared_output_grad                      + slice_offset;

    const int32_t* const input_index_row          = input_indices + block_idx * {max_active_indices};
    const {data_type}*   const input_value_row          = input_values  + block_idx * {max_active_indices};

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        shared_output_grad_slice[s] = (({compute_type})output_grad_slice[s]);
    }}

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        const {compute_type} sog = shared_output_grad_slice[s];
        if (sog != 0.0f)
        {{
            atomicAdd(&bias_grad_slice[s], ({data_type})sog);
        }}
    }}

    for (uint32_t k = 0; k < {max_active_indices}; ++k)
    {{
        const int32_t input_index = input_index_row[k];
        const {data_type}   input_value = input_value_row[k];
        if (input_index != -1)
        {{
            {data_type}* const weight_grad_slice = weight_grad + input_index * {output_size} + slice_offset;
            #pragma unroll
            for (int s = 0; s < {output_thread_slice_size}; ++s)
            {{
                const {compute_type} sog = shared_output_grad_slice[s];
                if (sog != 0.0f)
                {{
                    atomicAdd(&weight_grad_slice[s], ({data_type})(sog * (({compute_type})input_value)));
                }}
            }}
        }} else break;
    }}
}}

""".format(
                max_active_indices=max_active_indices,
                output_thread_slice_size=output_thread_slice_size,
                output_size=output_size,
                data_type=data_type,
                compute_type=compute_type,
            ),
            "sparse_input_linear_backward",
        )
        kernel.compile()
        _sparse_input_linear_backward_kernel_cache[key] = _kernel_with_threads(
            kernel, (num_threads,)
        )
    return _sparse_input_linear_backward_kernel_cache[key]
