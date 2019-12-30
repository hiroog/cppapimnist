// 2019/12/29 Hiroyuki Ogasawara
// vim:ts=4 sw=4 noet:

#include	<CNTKLibrary.h>
#include	<random>
#include	<iostream>
#include	"mnist_loader.h"


namespace {
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

CNTK::FunctionPtr	Dense( CNTK::Variable input_var, int output_size, const CNTK::DeviceDescriptor& device )
{
	size_t	input_size= input_var.Shape()[0];
	auto	weight= CNTK::Parameter(
							{ (size_t)output_size, (size_t)input_size },
							CNTK::DataType::Float,
							CNTK::HeNormalInitializer(),
							device
						);
	auto	bias= CNTK::Parameter(
							{ (size_t)output_size },
							CNTK::DataType::Float,
							0.0f,
							device
						);
	return	CNTK::Plus( bias, CNTK::Times( weight, input_var ) );
}


CNTK::FunctionPtr	Conv2D( CNTK::Variable input_var, int output_ch, int kernel_size, int stride, bool padding, const CNTK::DeviceDescriptor& device )
{
	size_t	input_ch= input_var.Shape()[2];
	auto	filter= CNTK::Parameter(
							{ (size_t)kernel_size, (size_t)kernel_size, input_ch, (size_t)output_ch },
							CNTK::DataType::Float,
							CNTK::HeNormalInitializer(),
							device
						);
	auto	bias= CNTK::Parameter(
							{ CNTK::NDShape::InferredDimension },
							CNTK::DataType::Float,
							0.0f,
							device
						);
	auto	y= CNTK::Convolution(
							filter,
							input_var,
							{ (size_t)stride, (size_t)stride, (size_t)input_ch },
							{ true },
							{ padding }
						);
	return	CNTK::Plus( bias, y );
}


CNTK::FunctionPtr	MaxPool2D( CNTK::Variable input_var, int kernel_size, int stride, bool padding )
{
	return	CNTK::Pooling(
					input_var,
					CNTK::PoolingType::Max,
					{ (size_t)kernel_size, (size_t)kernel_size },
					{ (size_t)stride, (size_t)stride },
					{ padding }
				);
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
}


class Model_MNist {
public:
	CNTK::FunctionPtr	PredictFunc;
	CNTK::Variable		InputVar;
public:
	Model_MNist( CNTK::DeviceDescriptor device )
	{
		CNTK::NDShape	flatten_shape( 1 );
		flatten_shape[0]= -1;

		InputVar= CNTK::InputVariable( { 28, 28, 1 }, CNTK::DataType::Float, L"xinput0" );

		auto	c0= CNTK::ReLU( Conv2D( InputVar, 16, 5, 1, false, device ) );
		auto	m0= MaxPool2D( c0, 2, 2, false );
		auto	d0= CNTK::Dropout( m0, 0.25f );
		auto	c1= CNTK::ReLU( Conv2D( d0, 32, 5, 1, false, device ) );
		auto	m1= MaxPool2D( c1, 2, 2, false );
		auto	d1= CNTK::Dropout( m1, 0.25f );
		auto	f= CNTK::Reshape( d1, flatten_shape );
		auto	f0= CNTK::ReLU( Dense( f, 128, device ) );
		auto	d2= CNTK::Dropout( f0, 0.5f );
		auto	f1= CNTK::ReLU( Dense( d2, 64, device ) );
		auto	d3= CNTK::Dropout( f1, 0.5f );
		auto	f3= Dense( d3, 10, device );
		PredictFunc= f3;
	}
};


static void	torch_test_train()
{
	constexpr int	EPOCH= 2;
	constexpr int	BATCH_SIZE= 32;
	constexpr int	DATA_COUNT= MNistLoader::TRAINDATA_SIZE;
	constexpr int	loop_count= DATA_COUNT / BATCH_SIZE;


	std::random_device	seed;
	std::mt19937	engine( seed() );
	std::uniform_int_distribution<int>	rand(0,DATA_COUNT-1);


	MNistLoader	loader( "../mnist" );

	auto	device= CNTK::DeviceDescriptor::UseDefaultDevice();
	Model_MNist	model( device );

	auto	label_var= CNTK::InputVariable( { 10 }, CNTK::DataType::Float, L"yinput" );
	auto	loss_func= CNTK::SquaredError( model.PredictFunc, label_var );
	auto	optimizer= CNTK::AdamLearner( model.PredictFunc->Parameters(), CNTK::LearningRateSchedule( 0.1f ), 0.9f );
	auto	trainer= CNTK::CreateTrainer( model.PredictFunc, loss_func, { optimizer } );


	std::vector<float>	image_input;
	std::vector<float>	label_input;
	image_input.resize( 28*28*BATCH_SIZE );
	label_input.resize( 10*BATCH_SIZE );

	std::unordered_map<CNTK::Variable,CNTK::MinibatchData>	input_map;

	for( int ei= 0 ; ei< EPOCH ; ei++ ){

		float	total_loss= 0.0f;
		for( int di= 0 ; di< loop_count ; di++ ){

			for( int bi= 0 ; bi< BATCH_SIZE ; bi++ ){
				int	ri= rand( engine );
				loader.GetFloatImage( &image_input[bi*28*28], ri );
				loader.GetFloatLabel10( &label_input[bi*10], ri );
			}

			input_map[model.InputVar]= CNTK::Value::CreateBatch( { 28, 28, 1 }, image_input, device, true );
			input_map[label_var]= CNTK::Value::CreateBatch( { 10 }, label_input, device, true );

			trainer->TrainMinibatch( input_map, device );

			total_loss+= static_cast<float>(trainer->PreviousMinibatchLossAverage());
		}
		std::cout << ei << " loss=" << total_loss / loop_count << '\n';
	}

	model.PredictFunc->Save( L"cpp_mnist_cntk.dnn" );
	std::cout << "ok\n";
}


static void	torch_test_predict()
{
	constexpr int	BATCH_SIZE= 128;
	constexpr int	DATA_COUNT= MNistLoader::TESTDATA_SIZE;
	constexpr int	loop_count= DATA_COUNT / BATCH_SIZE;

	MNistLoader	loader( "../mnist" );

	auto	device= CNTK::DeviceDescriptor::UseDefaultDevice();

	CNTK::FunctionPtr	PredictFunc= CNTK::Function::Load( L"cpp_mnist_cntk.dnn", device );
	auto	input_var= PredictFunc->Arguments()[0];
	auto	output_var= PredictFunc->Outputs()[0];

	std::vector<float>	image_input;
	image_input.resize( 28*28*BATCH_SIZE );
	int		index_table[BATCH_SIZE];

	std::unordered_map<CNTK::Variable,CNTK::ValuePtr>	input_map;
	std::unordered_map<CNTK::Variable,CNTK::ValuePtr>	output_map;

	int	data_index= 0;
	int	score= 0;
	for( int di= 0 ; di< loop_count ; di++ ){

		for( int bi= 0 ; bi< BATCH_SIZE ; bi++ ){
			index_table[bi]= data_index;
			loader.GetFloatImageTest( &image_input[bi*28*28], data_index );
			data_index++;
		}

		input_map[input_var]= CNTK::Value::CreateBatch( { 28, 28, 1 }, image_input, device, true );
		output_map[output_var]= nullptr;
		PredictFunc->Evaluate( input_map, output_map, device );

		const auto&	output_val= output_map[output_var];
		std::vector<std::vector<float>>	buffer;
		output_val->CopyVariableValueTo<float>( output_var, buffer );

		for( int bi= 0 ; bi< BATCH_SIZE ; bi++ ){
			float	max_val= 0.0f;
			int		max_index= -1;
			const float*	fptr= &buffer[bi][0];
			for( int fi= 0 ; fi< 10 ; fi++ ){
				float	fval= *fptr++;
				if( fval > max_val ){
					max_index= fi;
					max_val= fval;
				}
			}
			score+= (int)loader.GetLabelTest( index_table[bi] ) == max_index;
		}
	}
	std::cout << score * 100.0f / (loop_count*BATCH_SIZE) << " %\n";
	std::cout << "ok\n";
}


int main()
{
	torch_test_train();
	torch_test_predict();
	return	0;
}


