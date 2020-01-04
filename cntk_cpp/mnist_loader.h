// 2018/09/10 Hiroyuki Ogasawara
// vim:ts=4 sw=4 noet:

#ifndef	MNIST_LOADER_H_
#define	MNIST_LOADER_H_

#include	<cstdint>
#include	<vector>


class MNistLoader {
public:
	enum : unsigned int {
		IMAGE_WIDTH		=	28,
		IMAGE_HEIGHT	=	28,
		IMAGE_PIXSIZE	=	IMAGE_WIDTH * IMAGE_HEIGHT,
		LABEL_SIZE		=	10,
		TRAINDATA_SIZE	=	60000,
		TESTDATA_SIZE	=	10000,
	};
	struct ImageDataHeader {
		uint32_t	Magic;
		uint32_t	DataCount;
		uint32_t	Width;
		uint32_t	Height;
		uint8_t		Data[1];
	public:
		const uint8_t*	Get( int index ) const
		{
			return	&Data[ index * IMAGE_PIXSIZE ];
		}
	};
	struct LabelDataHeader {
		uint32_t	Magic;
		uint32_t	DataCount;
		uint8_t		Data[1];
	public:
		const uint8_t	Get( int index ) const
		{
			return	Data[ index ];
		}
	};
private:
	std::vector<void*>	ImageList;
public:
	MNistLoader( const char* mnist_data_path );
	~MNistLoader();
	void	Finalize();
	void	LoadImageFile( const char* mnist_data_path );
	const uint8_t*	GetImage( unsigned int index ) const;
	const uint8_t*	GetImageTest( unsigned int index ) const;
	const uint8_t	GetLabel( unsigned int index ) const;
	const uint8_t	GetLabelTest( unsigned int index ) const;
	void	GetFloatImage_( float* buffer, const uint8_t* image ) const;
	void	GetFloatLabel_( float* buffer, uint8_t label ) const;
	void	GetFloatImage( float* buffer, unsigned int index ) const;
	void	GetFloatLabel10( float* buffer, unsigned int index ) const;
	void	GetFloatImageTest( float* buffer, unsigned int index ) const;
};



#endif
