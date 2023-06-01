class Test_Classify:

    def test_torch_cuda_mem_get_info(self):
        import torch
        print(torch.cuda.mem_get_info())

    # def test_os_env_PYTORCH_CUDA_ALLOC_CONF(self):
    #     import os
    #     print(os.environ["PYTORCH_CUDA_ALLOC_CONF"])
