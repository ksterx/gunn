@ECHO OFF

pushd %~dp0

set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set BUILDDIR=_build

if "%1"=="" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR%

:end
popd
