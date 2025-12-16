import React, { useState } from 'react';
import { StyleSheet, View, Text, Keyboard, Alert } from 'react-native';
import { Searchbar, ActivityIndicator } from 'react-native-paper'; 

// REPLACE THIS with your computer's local IP address (keep port 8000)
const API_URL = 'http://127.0.0.1:8000';

export default function SearchScreen() {
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    // 1. Basic validation
    if (!searchQuery.trim()) return;

    Keyboard.dismiss();
    setLoading(true);

    try {
      // 2. Construct URL with query (q) and limit (k=3)
      // EncodeURIComponent ensures special characters don't break the URL
      const url = `${API_URL}/search?q=${encodeURIComponent(searchQuery)}&k=3`;

      // 3. Make the GET request
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }

      const data = await response.json();

      // 4. Log results to console (connect UI later)
      console.log('Search Results:', data);
      
    } catch (error) {
      console.error(error);
      Alert.alert('Error', 'Failed to fetch results. Check your connection.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={[styles.title, { color: 'black' }]}>Ilm Search</Text>
      <Text style={[styles.subtitle, { color: 'gray' }]}>Enter a topic or question</Text>
    
      <View style={styles.searchContainer}>
        <Searchbar
          placeholder="Search"
          onChangeText={setSearchQuery}
          value={searchQuery}
          // 5. Trigger search on 'Enter'
          onSubmitEditing={handleSearch}
          style={styles.searchBar}
          loading={loading} 
        />
        {/* Optional: Show loader below if needed */}
        {loading && <ActivityIndicator style={{ marginTop: 20 }} animating={true} />}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 10,
  },
  title: {
    fontSize: 30,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    marginBottom: 40,
  },
  searchContainer: {
    width: '100%',
    maxWidth: 400, 
    alignItems: 'center',
  },
  searchBar: {
    borderRadius: 15,
    backgroundColor: 'rgba(150, 150, 150, 0.1)',
  },
});