import torch
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def make_cuda_ext_mmdet3d(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)

if False:
    setup(
        name='det3d',
        ext_modules=[
                make_cuda_ext_mmdet3d(
                    name='furthest_point_sample_ext',
                    module='det3d.ops.furthest_point_sample',
                    sources=['src/furthest_point_sample.cpp'],
                    sources_cuda=['src/furthest_point_sample_cuda.cu']),
            ],
            cmdclass={'build_ext': BuildExtension})
else:
    setup(
        name='det3d',
        ext_modules=[
            CUDAExtension('ball_query_ext', [
                'src/ball_query.cpp',
                'src/ball_query_cuda.cu',
            ],
            define_macros=[('WITH_CUDA', None)],
            extra_compile_args={
                'cxx': [],
                'nvcc': [
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__',
            ]})],
            cmdclass={'build_ext': BuildExtension})
