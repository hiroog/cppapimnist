# 2019/12/29 Hiroyuki Ogasawara
# vim:ts=4 sw=4 et:

src_list= [
    'main.cpp',
    'mnist_loader.cpp',
]

cntk= tool.findPath( '../../cntk', 'CNTK_ROOT' )
#------------------------------------------------------------------------------

env= tool.createTargetEnvironment()

env.addLibraries( [ 'Cntk.Core-2.7', 'Cntk.Eval-2.7', ] )
env.addIncludePaths( [ os.path.join(cntk,'Include'), ] )
env.addLibPaths( [ os.path.join(cntk,'cntk'), ] )

#------------------------------------------------------------------------------
def makeExeName( env, src_name ):
    if src_name:
        return  env.getExeName( 'bin/' + src_name + '_' + env.getTargetArch() + '_' + env.getConfig() )
    return  '.'

env.EXE_NAME_FUNC= makeExeName


#------------------------------------------------------------------------------
task_list= []
for config in [ 'Release', 'Debug' ]:
    local_env= env.clone()
    local_env.setConfig( config )
    local_env.refresh()
    task_list.append( tool.addExeTask( local_env, 'mnist', src_list ) )

tool.addNamedTask( genv, 'build', task_list )


#------------------------------------------------------------------------------
def clean_files( env ):
    import BuildUtility
    BuildUtility.RemoveTree( 'obj' )

tool.addScriptTask( genv, 'clean', clean_files )



