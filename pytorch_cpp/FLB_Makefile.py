# 2019/12/29 Hiroyuki Ogasawara
# vim:ts=4 sw=4 et:

src_list= [
    'main.cpp',
    'mnist_loader.cpp',
]

ENABLE_CUDA=0
ENABLE_HIP=0    # ROCm

torch= tool.findPath( '../../libtorch', 'LIBTORCH_ROOT' )
#------------------------------------------------------------------------------

env= tool.createTargetEnvironment()

if env.getHostPlatform() == 'Windows':
    env.addCCFlags( ['-wd4146','-wd4244','-wd4018','-wd4251','-wd4275','-wd4267','-wd4522','-wd4273'] )
    torch_debug= tool.findPath( '../../libtorchdbg', 'LIBTORCH_ROOT' )
elif env.getHostPlatform() == 'Linux':
    env.addCCFlags( [ '-frtti', '-fexceptions', '-D_GLIBCXX_USE_CXX11_ABI=1', '-std=c++14', ] )
    torch_debug= torch

env.addLibraries( [ 'c10', 'torch', ] )

if ENABLE_CUDA:
    env.addLibraries( [ 'caffe2_nvrtc', 'c10_cuda', ] )
if ENABLE_HIP:
    env.addLibraries( [ 'c10_hip', 'torch_hip', ] )
if env.getHostPlatform() == 'Linux':
    env.addLibraries( [ 'torch_cpu', ] )

env.addIncludePaths( [
            os.path.join(torch,'include'),
            os.path.join(torch,'include/torch/csrc/api/include'),
        ] )

#------------------------------------------------------------------------------
def makeExeName( env, src_name ):
    if src_name:
        if env.getConfig() == 'Debug':
            return  env.getExeName( 'bindbg/' + src_name + '_' + env.getTargetArch() + '_' + env.getConfig() )
        else:
            return  env.getExeName( 'bin/' + src_name + '_' + env.getTargetArch() + '_' + env.getConfig() )
    return  '.'

env.EXE_NAME_FUNC= makeExeName


#------------------------------------------------------------------------------
task_list= []
for config in [ 'Release', 'Debug' ]:
    local_env= env.clone()
    local_env.setConfig( config )
    if config == 'Release':
        local_env.addLibPaths( [ os.path.join(torch,'lib'), ] )
    else:
        local_env.addLibPaths( [ os.path.join(torch_debug,'lib'), ] )
    local_env.refresh()
    task_list.append( tool.addExeTask( local_env, 'mnist', src_list ) )

tool.addNamedTask( genv, 'build', task_list )


#------------------------------------------------------------------------------
def clean_files( env ):
    import BuildUtility
    BuildUtility.RemoveTree( 'obj' )

tool.addScriptTask( genv, 'clean', clean_files )



