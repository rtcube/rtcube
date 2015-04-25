using namespace std;

namespace Generator{
	// Generates rows with random ranges (no file needed)
	void GenerateRandom(int no_rows, int no_cols);
	void GenerateRandom(int no_rows);
	// Generates rows based on the loaded file
	void Generate(int no_rows, bool with_time = true);
void GenerateToMemory(int no_rows, bool with_time  = true);
    void StartGenerating();

	bool LoadCubeFile(std::string filename);

	bool Connect(int port, std::string address);
	bool IsConnected();
	int Send(char * ip, int port);
	int Send();

	int NoColumns();
	void SetNoColumns(int no_cols);
}
