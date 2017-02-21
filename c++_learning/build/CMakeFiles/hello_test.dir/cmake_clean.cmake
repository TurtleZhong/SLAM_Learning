FILE(REMOVE_RECURSE
  "CMakeFiles/hello_test.dir/main.cpp.o"
  "hello_test.pdb"
  "hello_test"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang CXX)
  INCLUDE(CMakeFiles/hello_test.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
