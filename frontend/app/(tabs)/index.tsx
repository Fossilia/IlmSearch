import React, { useState } from 'react';
import { 
  View, 
  Text, 
  Keyboard, 
  Alert, 
  FlatList, 
  TouchableOpacity 
} from 'react-native';
import { Searchbar, ActivityIndicator, Card, Divider } from 'react-native-paper'; 

import styles from './styles'; 

const API_URL = 'http://127.0.0.1:8000';

// Define the shape of your data
type SearchResult = {
  reference: string;
  surah_name: string | null;
  verse_id: number | null;
  arabic: string | null;
  english: string | null;
  error: string | null;
};

export default function SearchScreen() {
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [hasSearched, setHasSearched] = useState(false); // Controls the layout shift
  const [activeTab, setActiveTab] = useState<'quran' | 'hadith'>('quran');

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    Keyboard.dismiss();
    setLoading(true);
    setHasSearched(true); // Triggers the "move up" animation effect
    setActiveTab('quran'); // Reset to default tab on new search

    try {
      const url = `${API_URL}/search?q=${encodeURIComponent(searchQuery)}&k=3`;
      const response = await fetch(url);
      
      if (!response.ok) throw new Error(`Error: ${response.status}`);

      const data = await response.json();
      setResults(data);
      
    } catch (error) {
      console.error(error);
      Alert.alert('Error', 'Failed to fetch results. Check your connection.');
    } finally {
      setLoading(false);
    }
  };

  // --- RENDER COMPONENT FOR LIST ITEMS ---
  const renderItem = ({ item }: { item: SearchResult }) => {
    // Handle error/missing content case
    if (item.error) {
      return (
        <Card style={[styles.card, styles.errorCard]}>
          <Card.Content>
            <Text style={styles.errorText}>⚠️ {item.reference}: {item.error}</Text>
          </Card.Content>
        </Card>
      );
    }

    // Regular Verse Card
    return (
      <Card style={styles.card}>
        <Card.Content>
          <View style={styles.cardHeader}>
            <Text style={styles.surahName}>{item.surah_name}</Text>
            <Text style={styles.verseRef}>{item.reference}</Text>
          </View>
          
          <Divider style={styles.divider} />
          
          <Text style={styles.arabicText}>{item.arabic}</Text>
          <Text style={styles.englishText}>{item.english}</Text>
        </Card.Content>
      </Card>
    );
  };

  return (
    <View style={[
      styles.container, 
      // Dynamically change alignment based on search state
      hasSearched ? { justifyContent: 'flex-start', paddingTop: 60 } : { justifyContent: 'center' }
    ]}>
      
      {/* Header Titles (Hide them after search to save space, or keep them small) */}
      {!hasSearched && (
        <View style={{ alignItems: 'center' }}>
          <Text style={[styles.title, { color: 'black' }]}>Ilm Search</Text>
          <Text style={[styles.subtitle, { color: 'gray' }]}>Enter a topic or question</Text>
        </View>
      )}

      {/* SEARCH BAR AREA */}
      <View style={styles.searchContainer}>
        <Searchbar
          placeholder="Search topics..."
          onChangeText={setSearchQuery}
          value={searchQuery}
          onSubmitEditing={handleSearch}
          style={styles.searchBar}
          loading={loading} 
        />
      </View>

      {/* TABS & RESULTS AREA (Only visible after search) */}
      {hasSearched && (
        <View style={styles.resultsArea}>
          
          {/* Custom Tab Selector */}
          <View style={styles.tabContainer}>
            <TouchableOpacity 
              style={[styles.tab, activeTab === 'quran' && styles.activeTab]}
              onPress={() => setActiveTab('quran')}
            >
              <Text style={[styles.tabText, activeTab === 'quran' && styles.activeTabText]}>Qur'an</Text>
            </TouchableOpacity>

            <TouchableOpacity 
              style={[styles.tab, activeTab === 'hadith' && styles.activeTab]}
              onPress={() => setActiveTab('hadith')}
            >
              <Text style={[styles.tabText, activeTab === 'hadith' && styles.activeTabText]}>Hadith</Text>
            </TouchableOpacity>
          </View>

          {/* CONTENT AREA */}
          {loading ? (
             <ActivityIndicator style={{ marginTop: 50 }} size="large" />
          ) : (
            <>
              {activeTab === 'quran' ? (
                <FlatList
                  data={results}
                  keyExtractor={(item, index) => item.reference + index}
                  renderItem={renderItem}
                  contentContainerStyle={styles.listContent}
                  showsVerticalScrollIndicator={false}
                />
              ) : (
                <View style={styles.placeholderContainer}>
                  <Text style={styles.placeholderText}>Hadith search coming soon...</Text>
                </View>
              )}
            </>
          )}
        </View>
      )}
    </View>
  );
};