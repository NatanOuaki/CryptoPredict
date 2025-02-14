import { useState, useEffect } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";
import "chart.js/auto";
import {
  Container,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  CssBaseline,
  ThemeProvider,
  createTheme,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  CircularProgress,
  TextField,
  Button,
} from "@mui/material";

// Dark mode theme setup
const getDesignTokens = (mode) => ({
  palette: {
    mode,
    ...(mode === "dark"
      ? {
          background: { default: "#121212", paper: "#1E1E1E" },
          text: { primary: "#ffffff" },
        }
      : {
          background: { default: "#ffffff", paper: "#f5f5f5" },
          text: { primary: "#000000" },
        }),
  },
});

const API_URL = "https://your-app-name.onrender.com"; // Replace with actual API URL
const COINGECKO_API = "https://api.coingecko.com/api/v3";

function App() {
  const [cryptos, setCryptos] = useState([]);
  const [selectedCrypto, setSelectedCrypto] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [themeMode, setThemeMode] = useState("dark");
  const [currentPrice, setCurrentPrice] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");

  const theme = createTheme(getDesignTokens(themeMode));

  useEffect(() => {
    fetchAllCryptos();
  }, []);

  const fetchAllCryptos = async () => {
    try {
      const response = await axios.get(`${COINGECKO_API}/coins/markets`, {
        params: {
          vs_currency: "usd",
          order: "market_cap_desc",
        },
      });
      setCryptos(response.data);
    } catch (error) {
      console.error("Error fetching cryptocurrencies:", error);
    }
  };

  const fetchPredictions = async (cryptoId) => {
    setLoading(true);
    await fetchRealTimePrice(cryptoId); // Fetch real-time price before predictions
    try {
      const response = await axios.get(`${API_URL}/predict/${cryptoId}/7`); // Default to 7 days for prediction
      setPredictions(response.data);
    } catch (error) {
      console.error("Error fetching predictions:", error);
    }
    setLoading(false);
  };

  const fetchRealTimePrice = async (cryptoId) => {
    try {
      const response = await axios.get(
        `${COINGECKO_API}/simple/price?ids=${cryptoId}&vs_currencies=usd`
      );
      setCurrentPrice(response.data[cryptoId].usd);
    } catch (error) {
      console.error("Error fetching real-time price:", error);
    }
  };

  const handleRowClick = (crypto) => {
    setSelectedCrypto(crypto);
    fetchPredictions(crypto.id);
  };

  const chartData = {
    labels: predictions.map((p) => new Date(p.date).toLocaleDateString()),
    datasets: [
      {
        label: `Predicted Price (${selectedCrypto?.name})`,
        data: predictions.map((p) => p.predicted_price),
        borderColor: "cyan",
        backgroundColor: "rgba(0, 255, 255, 0.2)",
        tension: 0.3,
      },
    ],
  };

  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
  };

  // Filter cryptos based on the search term
  const filteredCryptos = cryptos.filter((crypto) =>
    crypto.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="xl" style={{ textAlign: "center", padding: "20px", width: "100%" }}>
        <Typography variant="h4" gutterBottom>
          Crypto Price Predictions 🚀
        </Typography>

        <Paper elevation={3} style={{ padding: "20px", marginBottom: "20px" }}>
          <TextField
            label="Search Cryptos"
            variant="outlined"
            fullWidth
            value={searchTerm}
            onChange={handleSearchChange}
            style={{ marginBottom: "20px" }}
          />

          <Typography variant="h6">Available Cryptocurrencies</Typography>
          <TableContainer
            component={Paper}
            style={{
              marginTop: "20px",
              width: "100%", // Ensure table takes full width
              overflowX: "auto", // Enables horizontal scrolling if needed
            }}
          >
            <Table sx={{ minWidth: 650 }} style={{ width: "100%" }}>
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell>Price (USD)</TableCell>
                  <TableCell>Market Cap</TableCell>
                  <TableCell>24h Volume</TableCell>
                  <TableCell>24h Change</TableCell>
                  <TableCell>More Info</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredCryptos.map((crypto) => (
                  <TableRow key={crypto.id} hover onClick={() => handleRowClick(crypto)}>
                    <TableCell>{crypto.name}</TableCell>
                    <TableCell>${crypto.current_price.toLocaleString()}</TableCell>
                    <TableCell>${crypto.market_cap.toLocaleString()}</TableCell>
                    <TableCell>${crypto.total_volume.toLocaleString()}</TableCell>
                    <TableCell>{crypto.price_change_percentage_24h.toFixed(2)}%</TableCell>
                    <TableCell>
                      <Button variant="outlined">Details</Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>

        <Dialog open={Boolean(selectedCrypto)} onClose={() => setSelectedCrypto(null)}>
          <DialogTitle>Predictions for {selectedCrypto?.name}</DialogTitle>
          <DialogContent>
            {loading ? (
              <CircularProgress />
            ) : (
              predictions.length > 0 && <Line data={chartData} />
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setSelectedCrypto(null)} color="primary">
              Close
            </Button>
          </DialogActions>
        </Dialog>
      </Container>
    </ThemeProvider>
  );
}

export default App;
