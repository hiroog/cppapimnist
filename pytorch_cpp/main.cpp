// 2019/12/29 Hiroyuki Ogasawara
// vim:ts=4 sw=4 noet:

#include	<torch/torch.h>
#include	<random>
#include	"mnist_loader.h"


class Model_MNistImpl : public torch::nn::Module {
public:
	torch::nn::Conv2d	c0= nullptr;
	torch::nn::Conv2d	c1= nullptr;
	torch::nn::Linear	fc0= nullptr;
	torch::nn::Linear	fc1= nullptr;
	torch::nn::Linear	fc2= nullptr;
	torch::nn::Dropout	drop0= nullptr;
	torch::nn::Dropout	drop1= nullptr;
	torch::nn::Dropout	drop2= nullptr;
	torch::nn::Dropout	drop3= nullptr;
public:
	Model_MNistImpl()
	{
		c0= register_module( "c0", torch::nn::Conv2d( torch::nn::Conv2dOptions(  1, 16, 5 ) ) ); // in_ch out_ch kernel_size
		c1= register_module( "c1", torch::nn::Conv2d( torch::nn::Conv2dOptions( 16, 32, 5 ) ) );
		fc0= register_module( "fc0", torch::nn::Linear( 32*4*4, 128 ) );
		fc1= register_module( "fc1", torch::nn::Linear( 128, 64 ) );
		fc2= register_module( "fc2", torch::nn::Linear( 64, 10 ) );
		drop0= register_module( "drop0", torch::nn::Dropout( torch::nn::DropoutOptions(0.25f) ) );
		drop1= register_module( "drop1", torch::nn::Dropout( torch::nn::DropoutOptions(0.25f) ) );
		drop2= register_module( "drop2", torch::nn::Dropout( torch::nn::DropoutOptions(0.5f) ) );
		drop3= register_module( "drop3", torch::nn::Dropout( torch::nn::DropoutOptions(0.5f) ) );
	}

	torch::Tensor	forward( const torch::Tensor& input )
	{
		auto	x= torch::relu( c0( input ) );
				x= torch::max_pool2d( x, 2 );
				x= drop0( x );
				x= torch::relu( c1( x ) );
				x= torch::max_pool2d( x, 2 );
				x= drop1( x );
				x= x.view( {x.size(0), -1} );
				x= torch::relu( fc0( x ) );
				x= drop2( x );
				x= torch::relu( fc1( x ) );
				x= drop3( x );
				x= fc2( x );
		return	x;
	}
};


TORCH_MODULE( Model_MNist );


static void	torch_test_train()
{
	constexpr int	EPOCH= 2;
	constexpr int	BATCH_SIZE= 32;
	constexpr int	DATA_COUNT= MNistLoader::TRAINDATA_SIZE;
	constexpr int	loop_count= DATA_COUNT / BATCH_SIZE;


	std::random_device	seed;
	std::mt19937	engine( seed() );
	std::uniform_int_distribution<int>	rand(0,DATA_COUNT-1);

#if defined(VS_PROJECT)
	MNistLoader	loader( "../../mnist" );
#else
	MNistLoader	loader( "../mnist" );
#endif

	torch::Device	device= torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
	Model_MNist	model;
	model->to( device );
	torch::optim::Adam	optimizer( model->parameters(), torch::optim::AdamOptions( 0.001 ) );

	float	image_input[28*28*BATCH_SIZE];
	float	label_input[10*BATCH_SIZE];

	model->train();
	for( int ei= 0 ; ei< EPOCH ; ei++ ){

		float	total_loss= 0.0f;
		for( int di= 0 ; di< loop_count ; di++ ){

			for( int bi= 0 ; bi< BATCH_SIZE ; bi++ ){
				int	ri= rand( engine );
				loader.GetFloatImage( &image_input[bi*28*28], ri );
				loader.GetFloatLabel10( &label_input[bi*10], ri );
			}

			auto	input= torch::from_blob( image_input, { BATCH_SIZE, 1, 28, 28 }, torch::ScalarType::Float );
			auto	target= torch::from_blob( label_input, { BATCH_SIZE, 10 }, torch::ScalarType::Float );
			auto	input_gpu= input.to( device );
			auto	target_gpu= target.to( device );

			optimizer.zero_grad();

			auto	outputs= model->forward( input_gpu );
			auto	loss= torch::mse_loss( outputs, target_gpu );

			loss.backward();
			optimizer.step();

			auto	cur_loss= loss.item<float>();
			total_loss+= cur_loss;
		}
		std::cout << ei << " loss=" << total_loss / loop_count << '\n';
	}

	torch::save( model, "cpp_mnist_torch.pt" );
}


static void	torch_test_predict()
{
	constexpr int	BATCH_SIZE= 64;
	constexpr int	DATA_COUNT= MNistLoader::TESTDATA_SIZE;;
	constexpr int	loop_count= DATA_COUNT / BATCH_SIZE;

#if defined(VS_PROJECT)
	MNistLoader	loader( "../../mnist" );
#else
	MNistLoader	loader( "../mnist" );
#endif

	torch::Device	device= torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
	Model_MNist	model;
	model->to( device );

	torch::load( model, "cpp_mnist_torch.pt" );

	int		index_table[BATCH_SIZE];
	float	image_input[28*28*BATCH_SIZE];

	model->eval();

	int	data_index= 0;
	int	score= 0;
	for( int di= 0 ; di< loop_count ; di++ ){

		for( int bi= 0 ; bi< BATCH_SIZE ; bi++ ){
			index_table[bi]= data_index;
			loader.GetFloatImageTest( &image_input[bi*28*28], data_index );
			data_index++;
		}

		auto	input= torch::from_blob( image_input, { BATCH_SIZE, 1, 28, 28 }, torch::ScalarType::Float );
		auto	input_g= input.to( device );

		auto	outputs= model->forward( input_g );
		auto	outputs_cpu= outputs.to( torch::kCPU );

		auto	a= outputs_cpu.accessor<float,2>();
		for( int bi= 0 ; bi< BATCH_SIZE ; bi++ ){
			auto	b= a[bi];
			float	max_val= 0.0f;
			int		max_index= -1;
			for( int fi= 0 ; fi< 10 ; fi++ ){
				if( b[fi] > max_val ){
					max_index= fi;
					max_val= b[fi];
				}
			}
			score+= (int)loader.GetLabelTest( index_table[bi] ) == max_index;
		}
	}
	std::cout << score * 100.0f / (loop_count*BATCH_SIZE) << " %\n";
}


int main()
{
	torch_test_train();
	torch_test_predict();
	return	0;
}


