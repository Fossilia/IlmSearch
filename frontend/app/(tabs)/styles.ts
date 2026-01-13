import { StyleSheet } from 'react-native';

export default StyleSheet.create({
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
  gradeText: {
    fontSize: 14,
    color: '#666',
    fontStyle: 'italic',
    marginTop: 5,
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