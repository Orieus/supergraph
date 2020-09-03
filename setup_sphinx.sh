#!/bin/bash

source sphinx_settings/docs.env

mkdir $DOCSDIR

ROOT_DIR=$PWD

pushd $DOCSDIR

(echo y ; echo "$PROJECT_NAME" ; echo "$AUTHORS" ; echo $RELEASE_YEAR ; echo en)  | sphinx-quickstart

popd

ln -sf "$ROOT_DIR"/sphinx_settings/*.rst "$ROOT_DIR"/$DOCSDIR/source

pushd $DOCSDIR/source

patch -p0 < "$ROOT_DIR"/sphinx_settings/conf.patch

popd
