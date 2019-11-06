# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys,os,re
sys.path.insert(0, os.path.abspath('../../python'))

# -- Attempt to reformat Boost::Python docstrings to make them intelligible to Sphinx

def isBoostFunc(what,obj):
  return what=='function' and obj.__repr__().startswith('<Boost.Python.function object at 0x')

def isBoostMethod(what,obj):
  "I don't know how to distinguish boost and non-boost methods..."
  return what=='method' and obj.__repr__().startswith('<unbound method ')

def isBoostStaticMethod(what,obj):
  return what=='method' and obj.__repr__().startswith('<Boost.Python.function object at 0x')

def fixDocstring(app,what,name,obj,options,lines):
  if isBoostFunc(what,obj) or isBoostMethod(what,obj) or isBoostStaticMethod(what,obj):
    l2=boostFuncSignature(name,obj)[1]
    # we must replace lines one by one (in-place) :-|
    # knowing that l2 is always shorter than lines (l2 is docstring with the signature stripped off)
    for i in range(0,len(lines)):
      lines[i]=l2[i] if i<len(l2) else ''

def fixSignature(app, what, name, obj, options, signature, return_annotation):
  if what in ('attribute','class'): return signature,None
  elif isBoostFunc(what,obj):
    sig=boostFuncSignature(name,obj)[0] or ' (wrapped c++ function)'
    return sig,None
  elif isBoostMethod(what,obj):
    sig=boostFuncSignature(name,obj,removeSelf=True)[0]
    return sig,None
  elif isBoostStaticMethod(what,obj):
    sig=boostFuncSignature(name,obj,removeSelf=False)[0]+' [STATIC]'
    return sig,None

def boostFuncSignature(name,obj,removeSelf=False):
  """Scan docstring of obj, returning tuple of properly formatted boost python signature
  (first line of the docstring) and the rest of docstring (as list of lines).
  The rest of docstring is stripped of 4 leading spaces which are automatically
  added by boost.
  
  removeSelf will attempt to remove the first argument from the signature.
  """
  doc=obj.__doc__
  if doc==None: # not a boost method
    return None,None
  nname=name.split('.')[-1]
  docc=doc.split('\n')
  if len(docc)<2: return None,docc
  doc1=docc[1]
  # functions with weird docstring, likely not documented by boost
  if not re.match('^'+nname+r'(.*)->.*$',doc1):
    return None,docc
  if doc1.endswith(':'): doc1=doc1[:-1]
  strippedDoc=doc.split('\n')[2:]
  # check if all lines are padded
  allLinesHave4LeadingSpaces=True
  for l in strippedDoc:
    if l.startswith('    '): continue
    allLinesHave4LeadingSpaces=False; break
  # remove the padding if so
  if allLinesHave4LeadingSpaces: strippedDoc=[l[4:] for l in strippedDoc]
  for i in range(len(strippedDoc)):
    # fix signatures inside docstring (one function with multiple signatures)
    strippedDoc[i],n=re.subn(r'([a-zA-Z_][a-zA-Z0-9_]*\() \(object\)arg1(, |)',r'\1',strippedDoc[i].replace('->','→'))
  # inspect dosctring after mangling
  sig=doc1.split('(',1)[1]
  if removeSelf:
    # remove up to the first comma; if no comma present, then the method takes no arguments
    # if [ precedes the comma, add it to the result (ugly!)
    try:
      ss=sig.split(',',1)
      if ss[0].endswith('['): sig='['+ss[1]
      else: sig=ss[1]
    except IndexError:
      # grab the return value
      try:
       sig=') -> '+sig.split('->')[-1]
      except IndexError:
        sig=')'
  return '('+sig,strippedDoc

def setup(app):
  app.connect('autodoc-process-docstring',fixDocstring)
  app.connect('autodoc-process-signature',fixSignature)

# -- Project information -----------------------------------------------------

project = 'molgrid'
copyright = '2019, David Koes and Jocelyn Sunseri'
author = 'David Koes and Jocelyn Sunseri'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
