import React, { useState } from 'react';
import { StyleSheet, View, Text, Keyboard, TouchableWithoutFeedback } from 'react-native';
// Import the necessary components from React Native Paper
import { Searchbar, Provider as PaperProvider, DefaultTheme, MD3DarkTheme } from 'react-native-paper'; 

import Colors from '@/constants/Colors';
import { useColorScheme } from '@/components/useColorScheme';

// --- BEGIN index.tsx CONTENT ---

export const MyComponent = () => {
  const [searchQuery, setSearchQuery] = React.useState('');

  return (
    <Searchbar
      placeholder="Search"
      onChangeText={setSearchQuery}
      value={searchQuery}
    />
  );
};

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