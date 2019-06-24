#ifdef _WIN32
#define inline_macro __inline
#else
/*
* assume gcc for now
*/
#define inline_macro inline
#endif

#define private
