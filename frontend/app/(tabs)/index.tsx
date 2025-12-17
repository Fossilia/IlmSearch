import React, { useState } from 'react';
import { 
  StyleSheet, 
  View, 
  Text, 
  Keyboard, 
  Alert, 
  FlatList, 
  TouchableOpacity 
} from 'react-native';
import { Searchbar, ActivityIndicator, Card, Divider } from 'react-native-paper'; 

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

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    paddingHorizontal: 15,
    backgroundColor: '#F5F5F5', // Light grey background for better card contrast
  },
  // Typography
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    marginBottom: 30,
  },
  // Search Bar
  searchContainer: {
    width: '100%',
    maxWidth: 500, 
    zIndex: 1,
  },
  searchBar: {
    borderRadius: 12,
    backgroundColor: 'white',
    elevation: 4, // Android shadow
    shadowColor: '#000', // iOS shadow
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  // Tabs
  resultsArea: {
    flex: 1,
    width: '100%',
    marginTop: 20,
  },
  tabContainer: {
    flexDirection: 'row',
    marginBottom: 15,
    backgroundColor: '#E0E0E0',
    borderRadius: 10,
    padding: 4,
  },
  tab: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 8,
  },
  activeTab: {
    backgroundColor: 'white',
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  tabText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#757575',
  },
  activeTabText: {
    color: '#000',
  },
  // Results List & Cards
  listContent: {
    paddingBottom: 20,
  },
  card: {
    marginBottom: 15,
    backgroundColor: 'white',
    borderRadius: 12,
  },
  errorCard: {
    backgroundColor: '#FFF0F0',
    borderColor: '#FFCDD2',
    borderWidth: 1,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  surahName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2E7D32', // Islamic Green
  },
  verseRef: {
    fontSize: 14,
    color: '#666',
    fontWeight: '600',
  },
  divider: {
    marginVertical: 10,
  },
  arabicText: {
    fontSize: 22,
    textAlign: 'right', // Arabic is right-to-left
    writingDirection: 'rtl',
    marginBottom: 10,
    lineHeight: 38,
    fontFamily: 'System', // Use a standard system font, or a custom Arabic font if you have one
  },
  englishText: {
    fontSize: 16,
    lineHeight: 24,
    color: '#333',
  },
  errorText: {
    color: '#C62828',
    fontStyle: 'italic',
  },
  // Placeholders
  placeholderContainer: {
    flex: 1,
    alignItems: 'center',
    marginTop: 50,
  },
  placeholderText: {
    fontSize: 16,
    color: '#888',
    fontStyle: 'italic',
  },
});