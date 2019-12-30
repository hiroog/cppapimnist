// 2018/09/10 Hiroyuki Ogasawara
// vim:ts=4 sw=4 noet:

#include	"mnist_loader.h"
#include	<iostream>
#include	<fstream>
#include	<assert.h>
#include	<string.h>


template<typename T>
void zero_free( T*& ptr )
{
	if( ptr ){
		free( ptr );
		ptr= nullptr;
	}
}


MNistLoader::MNistLoader( const char* mnist_data_path )
{
	LoadImageFile( mnist_data_path );
}


MNistLoader::~MNistLoader()
{
	Finalize();
}


void	MNistLoader::Finalize()
{
	for( void* ptr : ImageList ){
		zero_free( ptr );
	}
	ImageList.clear();
}


static void*	LoadFileData( const char* file_name )
{
	std::ifstream	file( file_name, std::ios::binary|std::ios::ate );
	if( file ){
		size_t	size= file.tellg();
		void*	ptr= malloc( size );
		file.seekg( 0, std::ios_base::beg );
		file.read( (char*)ptr, size );
		return	ptr;
	}
	std::cerr << file_name << " not found\n";
	assert( 0 );
	return	nullptr;
}


void	MNistLoader::LoadImageFile( const char* mnist_data_path )
{
	Finalize();
	const char*	file_table[]= {
		"/train-images-idx3-ubyte",
		"/t10k-images-idx3-ubyte",
		"/train-labels-idx1-ubyte",
		"/t10k-labels-idx1-ubyte",
	};
	std::string	data_path= mnist_data_path;
	for( const auto* file_name : file_table ){
		auto	path= data_path + file_name;
		ImageList.push_back( LoadFileData( path.c_str() ) );
	}
}


template<typename ST>
auto	GetData( void* ptr, int index )
{
	return	reinterpret_cast<ST*>( ptr )->Get( index );
}


const uint8_t*	MNistLoader::GetImage( unsigned int index ) const
{
	assert( index < TRAINDATA_SIZE );
	return	GetData<ImageDataHeader>( ImageList[0], index );
}


const uint8_t*	MNistLoader::GetImageTest( unsigned int index ) const
{
	assert( index < TESTDATA_SIZE );
	return	GetData<ImageDataHeader>( ImageList[1], index );
}


const uint8_t	MNistLoader::GetLabel( unsigned int index ) const
{
	assert( index < TRAINDATA_SIZE );
	return	GetData<LabelDataHeader>( ImageList[2], index );
}


const uint8_t	MNistLoader::GetLabelTest( unsigned int index ) const
{
	assert( index < TESTDATA_SIZE );
	return	GetData<LabelDataHeader>( ImageList[3], index );
}


void	MNistLoader::GetFloatImage_( float* buffer, const uint8_t* image ) const
{
	auto*	ptr= image;
	float*	fptr= buffer;
	for( unsigned int pi= 0 ; pi< IMAGE_PIXSIZE ; pi++ ){
		*fptr++= *ptr++ * (1.0f/255.0f);
	}
}


void	MNistLoader::GetFloatLabel_( float* buffer, uint8_t label ) const
{
	assert( label < LABEL_SIZE );
	memset( buffer, 0, sizeof(float) * LABEL_SIZE );
	buffer[label]= 1.0f;
}


void	MNistLoader::GetFloatImage( float* buffer, unsigned int index ) const
{
	GetFloatImage_( buffer, GetImage( index ) );
}


void	MNistLoader::GetFloatLabel10( float* buffer, unsigned int index ) const
{
	GetFloatLabel_( buffer, GetLabel( index ) );
}


void	MNistLoader::GetFloatImageTest( float* buffer, unsigned int index ) const
{
	GetFloatImage_( buffer, GetImageTest( index ) );
}


