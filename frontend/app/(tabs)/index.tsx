import React, { useState } from 'react';
import { StyleSheet, View, Text, Keyboard, TouchableWithoutFeedback } from 'react-native';
// Import the necessary components from React Native Paper
import { Searchbar, Provider as PaperProvider, DefaultTheme, MD3DarkTheme } from 'react-native-paper'; 

import Colors from '@/constants/Colors';
import { useColorScheme } from '@/components/useColorScheme';

// --- BEGIN index.tsx CONTENT ---

export default function SearchScreen() {
  const [searchQuery, setSearchQuery] = useState('');
  const colorScheme = useColorScheme();
  
  // Define the maximum length for the user's query
  const MAX_QUERY_LENGTH = 50; 

  // Function to update the query while enforcing the character limit
  const onChangeSearch = (query: string) => {
    if (query.length <= MAX_QUERY_LENGTH) {
      setSearchQuery(query);
    }
  };

  // Function to handle the search submission (e.g., when the user presses Enter/Search button)
  const handleSearch = () => {
    // 💡 TO DO: Implement your API call here to fetch Qur'an and Hadith results
    console.log("Searching for:", searchQuery);
    Keyboard.dismiss();
    // 💡 TO DO: Navigate to the results screen after fetching data
  };

  // Set the theme based on the user's color scheme preference
  const theme = colorScheme === 'dark' ? MD3DarkTheme : DefaultTheme;
  const textColor = Colors[colorScheme ?? 'light'].text;

  return (
    // PaperProvider is required to use the React Native Paper components
    <PaperProvider theme={theme}>
      {/* Dismiss keyboard when tapping anywhere outside the input */}
      <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
        <View style={[styles.container, { backgroundColor: Colors[colorScheme ?? 'light'].background }]}>
          
          {/* Main Title */}
          <Text style={[styles.title, { color: textColor }]}>Qur'an & Hadith Search</Text>
          <Text style={[styles.subtitle, { color: 'gray' }]}>Enter a topic or keyword</Text>

          {/* Search Bar Component */}
          <View style={styles.searchContainer}>
            <Searchbar
              placeholder="e.g., 'Patience', 'Charity', 'Justice'"
              onChangeText={onChangeSearch}
              value={searchQuery}
              onSubmitEditing={handleSearch}
              elevation={2} // Adds a nice subtle shadow
              style={styles.searchBar}
              inputStyle={{ color: textColor }} 
              iconColor={Colors[colorScheme ?? 'light'].tint}
            />
            
            {/* Character Count Indicator */}
            <Text style={[styles.charCount, { color: 'gray' }]}>
              {searchQuery.length}/{MAX_QUERY_LENGTH}
            </Text>
          </View>

        </View>
      </TouchableWithoutFeedback>
    </PaperProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center', // This is key for vertical centering
    paddingHorizontal: 20,
  },
  title: {
    fontSize: 30,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    marginBottom: 40, // Space between subtitle and search bar
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
  charCount: {
    alignSelf: 'flex-end',
    marginTop: 8,
    fontSize: 12,
    marginRight: 5,
  },
});